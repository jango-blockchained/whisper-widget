"""Speech-to-text application using GTK 4 and Whisper."""

from __future__ import annotations

import os
import threading
import time
import wave
from typing import Any, Optional, Union, TypedDict

import gi  # type: ignore
gi.require_version('Gtk', '4.0')
from gi.repository import Gtk, Gio, GLib  # type: ignore # noqa: E402

import openai  # type: ignore # noqa: E402
import pyaudio  # type: ignore # noqa: E402
import pyperclip  # type: ignore # noqa: E402
import webrtcvad  # type: ignore # noqa: E402
from faster_whisper import WhisperModel  # type: ignore # noqa: E402
from pynput.keyboard import (  # type: ignore # noqa: E402
    Key, KeyCode, Listener, Controller
)

from whisper_widget.utils import noise_reduction  # noqa: E402


class OpenAIResponse(TypedDict):
    text: str


def check_microphone_access() -> bool:
    """Check if we can access the microphone."""
    try:
        p = pyaudio.PyAudio()
        p.get_device_count()  # Just to verify we can access devices
        default_input = p.get_default_input_device_info()
        print(f"Default input device: {default_input['name']}")
        p.terminate()
        print("Microphone access confirmed.")
        return True
    except Exception as e:
        print(f"Error accessing microphone: {e}")
        return False


class SpeechToTextApp:
    """Main application class for the speech-to-text widget."""

    def __init__(
        self,
        transcription_mode: str = "continuous",
        model_size: str = "base",
        language: str = "en",
        vad_sensitivity: int = 3,
        auto_detect_speech: bool = True,
        add_punctuation: bool = True,
        openai_api_key: Optional[str] = None,
    ) -> None:
        """Initialize the application."""
        # Create application
        self.app = Gtk.Application.new(
            "com.github.whisper-widget",
            Gio.ApplicationFlags.FLAGS_NONE
        )
        self.app.connect("activate", self.on_activate)

        # Initialize settings
        self.settings = {
            'transcription_mode': transcription_mode,
            'model_size': model_size,
            'language': language,
            'vad_sensitivity': vad_sensitivity,
            'auto_detect_speech': auto_detect_speech,
            'add_punctuation': add_punctuation,
            'min_speech_duration': 0.5,  # seconds
            'max_silence_duration': 1.0,  # seconds
            'min_audio_length': 1.0,  # seconds
            'speech_threshold': 0.5,  # 0.0-1.0
            'silence_threshold': 10,  # silence threshold
            'speech_start_chunks': 2,  # consecutive chunks to start
            'noise_reduce_threshold': 0.1,  # 0.0-1.0
            'openai_api_key': openai_api_key,
            'sample_rate': 16000,  # Fixed for Whisper
        }
        
        # Store instance attributes for compatibility with tests
        self.transcription_mode = transcription_mode
        self.model_size = model_size
        self.language = language
        self.vad_sensitivity = vad_sensitivity
        self.auto_detect_speech = auto_detect_speech
        self.add_punctuation = add_punctuation

        # Initialize Whisper model
        try:
            self.model = WhisperModel(
                model_size_or_path=self.model_size,
                device="cpu",
                compute_type="int8"
            )
        except Exception as e:
            print(f"Error initializing Whisper model: {e}")
            self.model = None

        # Initialize OpenAI if API key is provided
        if self.settings['openai_api_key']:
            openai.api_key = self.settings['openai_api_key']

        # Initialize audio components
        self.init_audio()

        # Initialize keyboard listener
        self.keyboard = Controller()
        self.listener = Listener(on_press=self.on_key_press)
        self.listener.start()

    def on_activate(self, app: Gtk.Application) -> None:
        """Called when the application is activated."""
        # Create window
        self.window = Gtk.ApplicationWindow.new(app)
        self.window.set_title("Speech to Text")
        self.window.set_default_size(300, 100)

        # Create header bar
        header = Gtk.HeaderBar()
        self.window.set_titlebar(header)

        # Create menu button
        menu_button = Gtk.MenuButton()
        header.pack_end(menu_button)

        # Create menu
        menu = Gio.Menu.new()
        self._create_menu(menu)
        menu_button.set_menu_model(menu)

        # Create main box
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.window.set_child(box)

        # Create status label
        self.status_label = Gtk.Label(label="Ready")
        box.append(self.status_label)

        # Create record button
        record_button = Gtk.Button(label="Record (F9)")
        record_button.connect("clicked", self._on_record_clicked)
        box.append(record_button)

        # Show window
        self.window.present()

    def _create_menu(self, menu: Gio.Menu) -> None:
        """Create the application menu."""
        # Settings section
        settings = Gio.Menu.new()
        
        # Transcription mode submenu
        mode_menu = Gio.Menu.new()
        mode_menu.append("Local", "app.set-mode::local")
        mode_menu.append("OpenAI", "app.set-mode::openai")
        settings.append_submenu("Transcription Mode", mode_menu)

        # Model size submenu
        model_menu = Gio.Menu.new()
        for size in ['tiny', 'base', 'small', 'medium', 'large']:
            model_menu.append(size.capitalize(), f"app.set-model::{size}")
        settings.append_submenu("Model Size", model_menu)

        # Language submenu
        lang_menu = Gio.Menu.new()
        languages = [
            ('English', 'en'),
            ('Spanish', 'es'),
            ('French', 'fr'),
            ('German', 'de'),
            ('Italian', 'it'),
            ('Japanese', 'ja'),
            ('Chinese', 'zh'),
            ('Auto', 'auto')
        ]
        for name, code in languages:
            lang_menu.append(name, f"app.set-language::{code}")
        settings.append_submenu("Language", lang_menu)

        # Add settings section
        menu.append_section(None, settings)

        # Add actions
        self._add_actions()

    def _add_actions(self) -> None:
        """Add actions for menu items."""
        # Mode action
        mode_action = Gio.SimpleAction.new_stateful(
            "set-mode",
            GLib.VariantType.new("s"),
            GLib.Variant.new_string(self.transcription_mode)
        )
        mode_action.connect("activate", self._on_mode_change)
        self.app.add_action(mode_action)

        # Model action
        model_action = Gio.SimpleAction.new_stateful(
            "set-model",
            GLib.VariantType.new("s"),
            GLib.Variant.new_string(self.model_size)
        )
        model_action.connect("activate", self._on_model_change)
        self.app.add_action(model_action)

        # Language action
        lang_action = Gio.SimpleAction.new_stateful(
            "set-language",
            GLib.VariantType.new("s"),
            GLib.Variant.new_string(self.language)
        )
        lang_action.connect("activate", self._on_language_change)
        self.app.add_action(lang_action)

        # Quit action
        quit_action = Gio.SimpleAction.new("quit", None)
        quit_action.connect("activate", self._on_quit)
        self.app.add_action(quit_action)

    def _on_record_clicked(self, button: Gtk.Button) -> None:
        """Handle record button click."""
        if not self.recording:
            self.start_recording()
        else:
            self.stop_recording()

    def _on_mode_change(
        self,
        action: Gio.SimpleAction,
        parameter: GLib.Variant
    ) -> None:
        """Handle transcription mode change."""
        mode = parameter.get_string()
        action.set_state(parameter)
        self.update_setting('transcription_mode', mode)

    def _on_model_change(
        self,
        action: Gio.SimpleAction,
        parameter: GLib.Variant
    ) -> None:
        """Handle model size change."""
        size = parameter.get_string()
        action.set_state(parameter)
        self.update_setting('model_size', size)

    def _on_language_change(
        self,
        action: Gio.SimpleAction,
        parameter: GLib.Variant
    ) -> None:
        """Handle language change."""
        lang = parameter.get_string()
        action.set_state(parameter)
        self.update_setting('language', lang)

    def _on_quit(self, action: Gio.SimpleAction, parameter: None) -> None:
        """Handle quit action."""
        self.quit(None)

    def update_status(self, status: str) -> None:
        """Update the status label."""
        self.status_label.set_text(status)

    def quit(self, _: Any) -> None:
        """Clean up and quit the application."""
        self.running = False
        if hasattr(self, 'audio_thread'):
            self.audio_thread.join()
        self.app.quit()

    def run(self) -> None:
        """Start the application."""
        # Start audio processing in a separate thread
        self.audio_thread = threading.Thread(
            target=self.audio_loop,
            daemon=True
        )
        self.audio_thread.start()
        
        # Run the application
        self.app.run(None)

    def init_audio(self) -> None:
        """Initialize audio components."""
        # Check microphone access first
        if not check_microphone_access():
            print("Error: Cannot access microphone. Please check permissions.")
            print("On Linux, you might need to:")
            print("1. Add your user to the 'audio' group:")
            print("   sudo usermod -a -G audio $USER")
            print("2. Log out and log back in")
            print("3. Check pulseaudio permissions")
            self.has_mic_access = False
        else:
            self.has_mic_access = True
            print("Microphone access confirmed.")

        # Buffer and state
        self.audio_buffer = bytearray()
        self.recording = False
        self.running = True
        self.computing = False

        # Only initialize audio if we have access
        if self.has_mic_access:
            # PyAudio configuration
            self.p = pyaudio.PyAudio()
            self.chunk_size = int(self.settings['sample_rate'] * 0.03)
            self.audio_format = pyaudio.paInt16
            self.channels = 1

            # Set up voice activity detection (VAD)
            self.vad = webrtcvad.Vad(self.settings['vad_sensitivity'])

    def update_setting(self, key: str, value: Any) -> None:
        """Update a setting and save to config."""
        self.settings[key] = value
        
        # Apply setting changes
        if key == 'transcription_mode':
            if value == 'local':
                self.model = WhisperModel(
                    self.settings['model_size'],
                    device="cpu",
                    compute_type="int8"
                )
            elif value == 'openai':
                openai.api_key = self.settings['openai_api_key']
        
        elif key == 'model_size' and self.settings['transcription_mode'] == 'local':
            self.model = WhisperModel(
                value,
                device="cpu",
                compute_type="int8"
            )
        
        elif key == 'vad_sensitivity':
            self.vad = webrtcvad.Vad(value)
        
        elif key == 'sample_rate':
            self.chunk_size = int(value * 0.03)
            # Restart audio thread with new sample rate
            if hasattr(self, 'audio_thread'):
                self.running = False
                self.audio_thread.join()
                self.running = True
                self.audio_thread = threading.Thread(
                    target=self.audio_loop,
                    daemon=True
                )
                self.audio_thread.start()

    # ------------------------------
    def on_key_press(
        self, 
        key: Optional[Union[Key, KeyCode]]
    ) -> Optional[bool]:
        """Toggle manual recording using the F9 key."""
        try:
            if key == Key.f9:
                if self.recording:
                    self.stop_recording()
                else:
                    self.start_recording()
        except Exception as e:
            print("Keyboard listener error:", e)
        return True

    # ------------------------------
    def start_recording(self) -> None:
        if not self.has_mic_access:
            print("Cannot start recording: No microphone access")
            return

        print("Manual recording started.")
        self.recording = True
        self.update_status('Recording')
        # Clear any previous audio data.
        self.audio_buffer = bytearray()

    def stop_recording(self) -> None:
        print("Manual recording stopped.")
        self.recording = False
        self.update_status('Ready')
        # If there is buffered audio, process it.
        if self.audio_buffer:
            self.process_audio_buffer()
            self.audio_buffer = bytearray()

    # ------------------------------
    def audio_loop(self) -> None:
        """Main audio processing loop."""
        stream = self.p.open(
            format=self.audio_format,
            channels=self.channels,
            rate=self.settings['sample_rate'],
            input=True,
            frames_per_buffer=self.chunk_size
        )
        
        # Initialize speech detection state
        silent_chunks = 0
        speech_chunks = 0
        recording_duration = 0.0
        silence_duration = 0.0
        chunk_duration = self.chunk_size / self.settings['sample_rate']

        while self.running:
            try:
                data = stream.read(
                    self.chunk_size, exception_on_overflow=False
                )
            except Exception as e:
                print("Audio stream error:", e)
                self.update_status('Error')
                continue

            # Apply noise reduction if enabled
            if self.settings['noise_reduce_threshold'] > 0:
                data = noise_reduction(
                    data, 
                    self.settings['sample_rate'],
                    threshold=self.settings['noise_reduce_threshold']
                )

            # Determine if the chunk contains speech
            is_speech = self.vad.is_speech(data, self.settings['sample_rate'])

            # Update speech/silence tracking
            if is_speech:
                speech_chunks += 1
                silence_duration = 0.0
            else:
                silence_duration += chunk_duration

            # Start recording if:
            # 1. Manual recording is active, or
            # 2. Auto-detect is on and we have enough speech chunks
            should_start = (
                self.recording or 
                (self.settings['auto_detect_speech'] and 
                 speech_chunks >= self.settings['speech_start_chunks'])
            )

            if should_start:
                if not self.recording:  # Auto-detection just started
                    print("Speech detected, starting recording...")
                    self.update_status('Recording')
                self.audio_buffer.extend(data)
                recording_duration += chunk_duration
                silent_chunks = 0 if is_speech else silent_chunks + 1
            else:
                speech_chunks = max(0, speech_chunks - 1)  # Decay speech chunks

            # Check if we should stop recording
            if self.audio_buffer:
                # Stop conditions:
                # 1. Too much silence
                # 2. Maximum silence duration reached
                # 3. Minimum speech duration met and silence detected
                should_stop = (
                    silent_chunks > self.settings['silence_threshold'] or
                    silence_duration >= self.settings['max_silence_duration'] or
                    (recording_duration >= self.settings['min_speech_duration'] and
                     silence_duration > 0.3)  # Small silence buffer
                )

                if should_stop:
                    # Only process if we meet minimum duration
                    if recording_duration >= self.settings['min_audio_length']:
                        print(f"Processing audio segment ({recording_duration:.1f}s)")
                        self.update_status('Computing')
                        self.process_audio_buffer()
                        self.update_status('Ready')
                    else:
                        print(f"Discarding short audio segment ({recording_duration:.1f}s)")

                    # Reset state
                    self.audio_buffer = bytearray()
                    recording_duration = 0.0
                    silence_duration = 0.0
                    speech_chunks = 0
                    silent_chunks = 0
                    
                    if self.recording:  # Was manual recording
                        self.recording = False
                        self.update_status('Ready')

            time.sleep(0.01)  # Small sleep to prevent CPU overuse
            
        stream.stop_stream()
        stream.close()

    # ------------------------------
    def process_audio_buffer(self) -> None:
        """Process the recorded audio buffer."""
        # Save the buffered audio to a temporary WAV file.
        temp_filename = "temp_audio.wav"
        wf = wave.open(temp_filename, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.p.get_sample_size(self.audio_format))
        wf.setframerate(self.settings['sample_rate'])
        wf.writeframes(self.audio_buffer)
        wf.close()

        # Transcribe the temporary file.
        transcription = self.transcribe_audio(temp_filename)
        if transcription:
            self.output_text(transcription)

        # Clean up temporary file.
        os.remove(temp_filename)

    def output_text(self, text: str) -> None:
        """Output the text either to clipboard or type it directly."""
        if self.settings['transcription_mode'] == 'clipboard':
            pyperclip.copy(text)
            print(f"Copied to clipboard: {text}")
        else:
            # Add a space before typing
            text = " " + text
            try:
                self.keyboard.type(text)
                print(f"Typed: {text}")
            except Exception as e:
                print(f"Error typing text: {e}")
                self.update_status('Error')

    # ------------------------------
    def transcribe_audio(self, audio_filename: str) -> str:
        """Transcribe the audio file using the selected method."""
        try:
            # Use instance attribute for transcription mode
            if self.transcription_mode == "local":
                # Use the local faster-whisper model
                if not self.model:
                    print("Error: Whisper model not initialized")
                    return ""
                    
                segments, info = self.model.transcribe(
                    audio_filename,
                    beam_size=5,
                    word_timestamps=True,
                    language=self.language,
                    condition_on_previous_text=True,
                    no_speech_threshold=0.6
                )
                
                # Join all segment texts
                text = " ".join([seg.text for seg in segments]).strip()
                
                # Add basic punctuation if enabled
                if text and self.add_punctuation and not text[-1] in '.!?':
                    text += '.'
                    
                return text
                
            elif self.transcription_mode == "openai":
                # Use OpenAI's API
                if not self.settings.get('openai_api_key'):
                    print("Error: OpenAI API key not set")
                    return ""
                    
                with open(audio_filename, "rb") as audio_file:
                    result: OpenAIResponse = openai.Audio.transcribe(
                        "whisper-1", 
                        audio_file
                    )
                    return result["text"].strip()
            else:
                print(
                    f"Error: Invalid transcription mode "
                    f"'{self.transcription_mode}'"
                )
                return ""
                
        except Exception as e:
            print(f"Transcription error: {e}")
            self.update_status('Error')
            return ""


# ------------------------------
if __name__ == "__main__":
    app = SpeechToTextApp()
    app.run() 