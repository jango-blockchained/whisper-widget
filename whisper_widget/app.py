"""Speech-to-text application using GTK 4 and Whisper."""

from __future__ import annotations

import os
import threading
import time
import wave
from datetime import datetime
from typing import Any, Optional, Union, TypedDict

import gi  # type: ignore
gi.require_version('Gtk', '4.0')
from gi.repository import (  # type: ignore # noqa: E402
    Gtk, Gio, GLib
)

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


def print_status_header() -> None:
    """Print the status header with column names."""
    print("\n" + "=" * 80)
    header = "Time         | Status     | Duration   | Mode     | Details"
    print(header)
    print("-" * 80)


def print_status_line(
    status: str,
    duration: float = 0.0,
    mode: str = "",
    details: str = ""
) -> None:
    """Print a status line with current time and information."""
    current_time = datetime.now().strftime("%H:%M:%S")
    duration_str = f"{duration:.1f}s" if duration > 0 else ""
    line = f"{current_time:12} | {status:10} | {duration_str:10} | {mode:8} | {details}"
    print(line)


def check_microphone_access() -> bool:
    """Check if we have access to the microphone."""
    try:
        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=1024
        )
        stream.stop_stream()
        stream.close()
        p.terminate()
        return True
    except OSError:
        return False


class SpeechToTextApp:
    """Main application class."""

    def __init__(
        self,
        transcription_mode: str = "local",
        output_mode: str = "continuous",
        model_size: str = "base",
        language: str = "en",
        vad_sensitivity: int = 3,
        auto_detect_speech: bool = True,
        add_punctuation: bool = True,
        openai_api_key: Optional[str] = None,
    ) -> None:
        """Initialize the application."""
        # Print initial header
        print("\nWhisper Widget - Speech-to-Text Application")
        print("==========================================")
        print(f"Transcription Mode: {transcription_mode}")
        print(f"Output Mode: {output_mode}")
        print(f"Model Size: {model_size}")
        print(f"Language: {language}")
        print(f"VAD Sensitivity: {vad_sensitivity}")
        print(f"Auto-detect Speech: {auto_detect_speech}")
        print(f"Add Punctuation: {add_punctuation}")
        print("==========================================\n")
        
        print_status_header()
        
        # Create application
        self.app = Gtk.Application.new(
            "com.github.whisper-widget",
            Gio.ApplicationFlags.FLAGS_NONE
        )
        self.app.connect("activate", self.on_activate)
        
        # Initialize window as None - will be created in on_activate
        self.window: Optional[Gtk.ApplicationWindow] = None
        
        # Create status label
        self.status_label = Gtk.Label()
        self.status_label.set_text("Ready")
        
        # Initialize settings
        self.transcription_mode = transcription_mode
        self.output_mode = output_mode
        self.model_size = model_size
        self.language = language
        self.vad_sensitivity = vad_sensitivity
        self.auto_detect_speech = auto_detect_speech
        self.add_punctuation = add_punctuation
        self.openai_api_key = openai_api_key
        
        # Initialize state variables
        self.is_recording = False
        self.audio_thread: Optional[threading.Thread] = None
        self.keyboard_listener: Optional[Listener] = None
        self.keyboard = Controller()
        self.recording_start_time: Optional[float] = None
        
        # Set default audio parameters
        self.sample_rate = 16000
        self.min_speech_duration = 0.2
        self.max_silence_duration = 0.5
        self.min_audio_length = 0.3
        self.speech_threshold = 0.5
        self.silence_threshold = 5
        self.speech_start_chunks = 1
        self.noise_reduce_threshold = 0.1
        
        if openai_api_key:
            openai.api_key = openai_api_key
        
        # Initialize audio components
        self.init_audio()

    def update_status(self, status: str) -> None:
        """Update the status label and print status line."""
        self.status_label.set_text(status)
        duration = 0.0
        if self.recording_start_time and self.is_recording:
            duration = time.time() - self.recording_start_time
        
        details = []
        if self.transcription_mode == "local":
            details.append(f"Model: {self.model_size}")
        if self.auto_detect_speech:
            details.append("Auto-detect")
        if self.add_punctuation:
            details.append("Punctuation")
            
        print_status_line(
            status,
            duration,
            self.transcription_mode,
            ", ".join(details)
        )

    def on_activate(self, app: Gtk.Application) -> None:
        """Handle application activation."""
        # Create the main window
        self.window = Gtk.ApplicationWindow.new(app)
        self.window.set_title("Whisper Widget")
        self.window.set_default_size(400, 100)
        
        # Create header bar
        header = Gtk.HeaderBar()
        self.window.set_titlebar(header)
        
        # Create menu button
        menu_button = Gtk.MenuButton()
        header.pack_end(menu_button)
        
        # Create menu model
        menu = Gio.Menu.new()
        
        # Add transcription mode submenu
        trans_menu = Gio.Menu.new()
        for mode in ['local', 'openai']:
            action = f'app.trans_mode_{mode}'
            self.app.add_action(
                Gio.SimpleAction.new_stateful(
                    f'trans_mode_{mode}',
                    None,
                    GLib.Variant.new_boolean(self.transcription_mode == mode)
                )
            )
            trans_menu.append(mode.capitalize(), action)
        menu.append_submenu('Transcription Mode', trans_menu)
        
        # Add output mode submenu
        output_menu = Gio.Menu.new()
        for mode in ['continuous', 'clipboard']:
            action = f'app.output_mode_{mode}'
            self.app.add_action(
                Gio.SimpleAction.new_stateful(
                    f'output_mode_{mode}',
                    None,
                    GLib.Variant.new_boolean(self.output_mode == mode)
                )
            )
            output_menu.append(mode.capitalize(), action)
        menu.append_submenu('Output Mode', output_menu)
        
        # Add model size submenu
        model_menu = Gio.Menu.new()
        for size in ['tiny', 'base', 'small', 'medium', 'large']:
            action = f'app.model_size_{size}'
            self.app.add_action(
                Gio.SimpleAction.new_stateful(
                    f'model_size_{size}',
                    None,
                    GLib.Variant.new_boolean(self.model_size == size)
                )
            )
            model_menu.append(size.capitalize(), action)
        menu.append_submenu('Model Size', model_menu)
        
        # Add language submenu
        lang_menu = Gio.Menu.new()
        languages = [
            ('en', 'English'),
            ('fr', 'French'),
            ('de', 'German'),
            ('es', 'Spanish'),
            ('it', 'Italian'),
            ('pt', 'Portuguese'),
            ('nl', 'Dutch'),
            ('pl', 'Polish'),
            ('ru', 'Russian'),
            ('zh', 'Chinese'),
            ('ja', 'Japanese'),
            ('ko', 'Korean'),
        ]
        for code, name in languages:
            action = f'app.lang_{code}'
            self.app.add_action(
                Gio.SimpleAction.new_stateful(
                    f'lang_{code}',
                    None,
                    GLib.Variant.new_boolean(self.language == code)
                )
            )
            lang_menu.append(name, action)
        menu.append_submenu('Language', lang_menu)
        
        menu.append('Auto-detect Speech', 'app.auto_detect')
        self.app.add_action(
            Gio.SimpleAction.new_stateful(
                'auto_detect',
                None,
                GLib.Variant.new_boolean(self.auto_detect_speech)
            )
        )
        
        menu.append('Add Punctuation', 'app.add_punct')
        self.app.add_action(
            Gio.SimpleAction.new_stateful(
                'add_punct',
                None,
                GLib.Variant.new_boolean(self.add_punctuation)
            )
        )
        
        menu.append('Quit', 'app.quit')
        quit_action = Gio.SimpleAction.new('quit', None)
        quit_action.connect('activate', self.quit)
        self.app.add_action(quit_action)
        
        # Set menu model
        menu_button.set_menu_model(menu)
        
        # Create a vertical box for layout
        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.window.set_child(vbox)
        
        # Add status label to window
        vbox.append(self.status_label)
        
        # Create record button
        record_button = Gtk.Button(label="Record")
        record_button.connect("clicked", self._on_record_clicked)
        vbox.append(record_button)
        
        # Show the window
        self.window.present()

    def _on_trans_mode_change(self, action: Gio.SimpleAction, parameter: None) -> None:
        """Handle transcription mode change."""
        mode = action.get_name().replace('trans_mode_', '')
        self.transcription_mode = mode
        
        # Update model availability
        if mode == 'local':
            try:
                self.model = WhisperModel(
                    self.model_size,
                    device="cpu",
                    compute_type="int8"
                )
            except Exception as e:
                print(f"Error initializing Whisper model: {e}")
                self.model = None
        else:  # openai
            if self.openai_api_key:
                openai.api_key = self.openai_api_key
            else:
                print("Warning: OpenAI API key not set")

    def _on_output_mode_change(self, action: Gio.SimpleAction, parameter: None) -> None:
        """Handle output mode change."""
        mode = action.get_name().replace('output_mode_', '')
        self.output_mode = mode

    def _on_model_change(self, action: Gio.SimpleAction, parameter: None) -> None:
        """Handle model size change."""
        size = action.get_name().replace('model_size_', '')
        self.model_size = size
        if self.transcription_mode == 'local':
            try:
                self.model = WhisperModel(
                    size,
                    device="cpu",
                    compute_type="int8"
                )
            except Exception as e:
                print(f"Error initializing Whisper model: {e}")
                self.model = None

    def _on_language_change(self, action: Gio.SimpleAction, parameter: None) -> None:
        """Handle language change."""
        code = action.get_name().replace('lang_', '')
        self.language = code

    def _on_auto_detect_toggle(self, action: Gio.SimpleAction, parameter: None) -> None:
        """Handle auto-detect speech toggle."""
        state = not action.get_state()
        action.set_state(GLib.Variant.new_boolean(state))
        self.auto_detect_speech = state

    def _on_punctuation_toggle(self, action: Gio.SimpleAction, parameter: None) -> None:
        """Handle add punctuation toggle."""
        state = not action.get_state()
        action.set_state(GLib.Variant.new_boolean(state))
        self.add_punctuation = state

    def _on_record_clicked(self, button: Gtk.Button) -> None:
        """Handle record button click."""
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()

    def quit(self, _: Any) -> None:
        """Clean up and quit the application."""
        self.is_recording = False
        if self.audio_thread:
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
            self.update_status('No Mic')
        else:
            self.has_mic_access = True
            print("Microphone access confirmed.")
            self.update_status('Ready')

        # Buffer and state
        self.audio_buffer = bytearray()
        self.is_recording = False
        self.running = True
        self.computing = False

        # Only initialize audio if we have access
        if self.has_mic_access:
            # PyAudio configuration
            self.p = pyaudio.PyAudio()
            self.chunk_size = int(16000 * 0.03)
            self.audio_format = pyaudio.paInt16
            self.channels = 1

            # Set up voice activity detection (VAD)
            self.vad = webrtcvad.Vad(self.vad_sensitivity)

    def update_setting(self, key: str, value: Any) -> None:
        """Update a setting and save to config."""
        if key == 'transcription_mode':
            self.transcription_mode = value
        elif key == 'output_mode':
            self.output_mode = value
        elif key == 'model_size':
            self.model_size = value
        elif key == 'language':
            self.language = value
        elif key == 'vad_sensitivity':
            self.vad = webrtcvad.Vad(value)
        elif key == 'sample_rate':
            self.chunk_size = int(value * 0.03)
            # Restart audio thread with new sample rate
            if self.audio_thread:
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
                if self.is_recording:
                    self.stop_recording()
                else:
                    self.start_recording()
        except Exception as e:
            print("Keyboard listener error:", e)
            self.update_status('Error')
        return True

    # ------------------------------
    def start_recording(self) -> None:
        if not self.has_mic_access:
            print_status_line("Error", details="No microphone access")
            self.update_status('No Mic')
            return

        self.recording_start_time = time.time()
        self.is_recording = True
        self.update_status('Recording')
        # Clear any previous audio data
        self.audio_buffer = bytearray()

    def stop_recording(self) -> None:
        duration = 0.0
        if self.recording_start_time:
            duration = time.time() - self.recording_start_time
        print_status_line("Stopped", duration)
        
        self.is_recording = False
        self.recording_start_time = None
        self.update_status('Ready')
        # If there is buffered audio, process it
        if self.audio_buffer:
            self.process_audio_buffer()
            self.audio_buffer = bytearray()

    # ------------------------------
    def audio_loop(self) -> None:
        """Main audio processing loop."""
        if not self.has_mic_access:
            return

        stream = self.p.open(
            format=self.audio_format,
            channels=self.channels,
            rate=16000,
            input=True,
            frames_per_buffer=self.chunk_size
        )
        
        # Initialize speech detection state
        silent_chunks = 0
        speech_chunks = 0
        recording_duration = 0.0
        silence_duration = 0.0
        chunk_duration = self.chunk_size / 16000

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
            if self.noise_reduce_threshold > 0:
                data = noise_reduction(
                    data, 
                    16000,
                    threshold=self.noise_reduce_threshold
                )

            # Determine if the chunk contains speech
            is_speech = self.vad.is_speech(data, 16000)

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
                self.is_recording or 
                (self.auto_detect_speech and 
                 speech_chunks >= self.speech_start_chunks)
            )

            if should_start:
                if not self.is_recording:  # Auto-detection just started
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
                    silent_chunks > self.silence_threshold or
                    silence_duration >= self.max_silence_duration or
                    (recording_duration >= self.min_speech_duration and
                     silence_duration > 0.3)  # Small silence buffer
                )

                if should_stop:
                    # Only process if we meet minimum duration
                    if recording_duration >= self.min_audio_length:
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
                    
                    if self.is_recording:  # Was manual recording
                        self.is_recording = False
                        self.update_status('Ready')

            time.sleep(0.01)  # Small sleep to prevent CPU overuse
            
        stream.stop_stream()
        stream.close()

    # ------------------------------
    def process_audio_buffer(self) -> None:
        """Process the recorded audio buffer."""
        # Save the buffered audio to a temporary WAV file
        temp_filename = "temp_audio.wav"
        wf = wave.open(temp_filename, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.p.get_sample_size(self.audio_format))
        wf.setframerate(16000)
        wf.writeframes(self.audio_buffer)
        wf.close()

        details = [
            f"Size: {len(self.audio_buffer)/16000:.1f}s",
            f"Lang: {self.language}"
        ]
        print_status_line(
            "Processing",
            mode=self.transcription_mode,
            details=", ".join(details)
        )
        
        # Transcribe the temporary file
        transcription = self.transcribe_audio(temp_filename)
        if transcription:
            print_status_line(
                "Transcribed",
                details=f"Length: {len(transcription)} chars"
            )
            self.output_text(transcription)
        else:
            print_status_line("Failed", details="Transcription error")

        # Clean up temporary file
        os.remove(temp_filename)

    def output_text(self, text: str) -> None:
        """Output the text either to clipboard or type it directly."""
        if self.output_mode == 'clipboard':
            pyperclip.copy(text)
            print_status_line(
                "Output",
                mode="clipboard",
                details=f"Text: {text[:50]}..."
            )
        else:  # continuous
            # Add a space before typing
            text = " " + text
            try:
                self.keyboard.type(text)
                print_status_line(
                    "Output",
                    mode="typing",
                    details=f"Text: {text[:50]}..."
                )
            except Exception as e:
                error_msg = f"Typing error: {str(e)}"
                print_status_line("Error", details=error_msg)
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
                if not self.openai_api_key:
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