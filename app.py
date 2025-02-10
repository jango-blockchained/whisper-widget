from __future__ import annotations

import json
import os
import threading
import time
import wave
from pathlib import Path
from typing import Any, Dict, Optional, Union, Callable, cast, TypedDict

import gi
gi.require_version('Gtk', '3.0')
gi.require_version('AppIndicator3', '0.1')
from gi.repository import Gtk, AppIndicator3

import openai
import pyaudio
import pyperclip
import webrtcvad
from faster_whisper import WhisperModel
from pynput.keyboard import Key, KeyCode, Listener, Controller


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


# ------------------------------
# Optional noise cancellation stub.
# Replace or extend with a real algorithm or library (e.g., noisereduce)
def noise_reduction(
    audio_data: bytes,
    sample_rate: int,
    threshold: float = 0.1
) -> bytes:
    """
    Apply noise reduction to the audio data.
    
    Args:
        audio_data: Raw audio data bytes
        sample_rate: Audio sample rate in Hz
        threshold: Noise reduction sensitivity (0.0-1.0)
        
    Returns:
        Processed audio data bytes
    """
    # For a basic implementation, we could:
    # 1. Convert bytes to numpy array
    # 2. Apply a simple noise gate
    # 3. Convert back to bytes
    # 
    # Example with numpy:
    # import numpy as np
    # audio_array = np.frombuffer(audio_data, dtype=np.int16)
    # max_amplitude = np.max(np.abs(audio_array))
    # noise_gate = max_amplitude * threshold
    # audio_array[np.abs(audio_array) < noise_gate] = 0
    # return audio_array.tobytes()
    
    # For now, simply return the data unmodified
    return audio_data


def load_settings() -> Dict[str, Any]:
    """Load settings from config file."""
    config_dir = Path.home() / '.config' / 'whisper-widget'
    config_file = config_dir / 'config.json'
    default_settings: Dict[str, Any] = {
        'transcription_mode': 'continuous',  # or 'clipboard'
        'model_size': 'base',  # tiny, base, small, medium, large
        'language': 'en',  # Default to English instead of auto
        'vad_sensitivity': 3,  # 1-3
        'auto_detect_speech': True,
        'add_punctuation': True,
        'sample_rate': 16000,
        # Speech detection parameters
        'min_speech_duration': 0.5,  # Minimum duration (seconds) to consider as speech
        'max_silence_duration': 1.0,  # Maximum silence duration (seconds) before stopping
        'min_audio_length': 1.0,  # Minimum audio segment length (seconds) to process
        'speech_threshold': 0.5,  # Speech probability threshold (0.0-1.0)
        'silence_threshold': 10,  # Number of silent chunks before stopping
        'speech_start_chunks': 2,  # Number of speech chunks to trigger recording
        'noise_reduce_threshold': 0.1,  # Noise reduction sensitivity (0.0-1.0)
    }
    
    if config_file.exists():
        try:
            with open(config_file) as f:
                return {**default_settings, **json.load(f)}
        except Exception as e:
            print(f"Error loading settings: {e}")
            return default_settings
    return default_settings


def save_settings(settings: Dict[str, Any]) -> None:
    """Save settings to config file."""
    config_dir = Path.home() / '.config' / 'whisper-widget'
    config_file = config_dir / 'config.json'
    config_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(config_file, 'w') as f:
            json.dump(settings, f, indent=4)
    except Exception as e:
        print(f"Error saving settings: {e}")


# ------------------------------
class SpeechToTextApp:
    # Icon colors for different states
    ICON_COLORS = {
        'ready': (0, 128, 0),        # Green
        'recording': (255, 0, 0),     # Red
        'computing': (255, 165, 0),   # Orange
        'error': (128, 0, 0),         # Dark red
        'no_mic': (128, 128, 128)     # Gray - no microphone access
    }

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
        # Load saved settings or use defaults
        self.settings = load_settings()
        
        # Store instance attributes for compatibility with tests
        self.transcription_mode = transcription_mode
        self.model_size = model_size
        self.language = language
        self.vad_sensitivity = vad_sensitivity
        self.auto_detect_speech = auto_detect_speech
        self.add_punctuation = add_punctuation
        
        # Update settings with constructor parameters
        self.settings.update({
            'transcription_mode': transcription_mode,
            'model_size': model_size,
            'language': language,
            'vad_sensitivity': vad_sensitivity,
            'auto_detect_speech': auto_detect_speech,
            'add_punctuation': add_punctuation,
            'openai_api_key': openai_api_key,
            'sample_rate': 16000,  # Fixed for Whisper
        })
        
        # Save updated settings
        save_settings(self.settings)

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

        # Create menu
        self.menu = Gtk.Menu()
        self._create_menu()  # Call private method to create menu items
        
        # Initialize system tray icon
        self.indicator = AppIndicator3.Indicator.new(
            "speech-to-text",
            "audio-input-microphone",
            AppIndicator3.IndicatorCategory.APPLICATION_STATUS
        )
        self.indicator.set_status(AppIndicator3.IndicatorStatus.ACTIVE)
        self.indicator.set_menu(self.menu)

        # Initialize audio components
        self.init_audio()

        # Initialize keyboard listener
        self.keyboard = Controller()
        self.listener = Listener(on_press=self.on_key_press)
        self.listener.start()

    def _create_menu(self) -> None:
        """Create the menu items for the system tray icon."""
        # Create main menu
        self.menu = Gtk.Menu()

        # Settings submenu
        settings_item = Gtk.MenuItem(label="Settings")
        settings_menu = Gtk.Menu()
        settings_item.set_submenu(settings_menu)

        # 1. Transcription mode menu
        mode_item = Gtk.MenuItem(label="Transcription Mode")
        mode_menu = Gtk.Menu()
        mode_item.set_submenu(mode_menu)

        # Local mode
        local_mode = Gtk.RadioMenuItem(label="Local")
        local_mode.connect(
            'activate',
            lambda w: (
                self.update_setting('transcription_mode', 'local')
                if w.get_active() else None
            )
        )
        local_mode.set_active(self.transcription_mode == 'local')
        mode_menu.append(local_mode)

        # OpenAI mode
        openai_mode = Gtk.RadioMenuItem.new_with_label_from_widget(local_mode, "OpenAI")
        openai_mode.connect(
            'activate',
            lambda w: (
                self.update_setting('transcription_mode', 'openai')
                if w.get_active() else None
            )
        )
        openai_mode.set_active(self.transcription_mode == 'openai')
        mode_menu.append(openai_mode)

        settings_menu.append(mode_item)

        # 2. Model size menu
        model_item = Gtk.MenuItem(label="Model Size")
        model_menu = Gtk.Menu()
        model_item.set_submenu(model_menu)

        model_sizes = ['tiny', 'base', 'small', 'medium', 'large']
        model_group = None
        for size in model_sizes:
            if model_group is None:
                model_radio = Gtk.RadioMenuItem(label=size)
                model_group = model_radio
            else:
                model_radio = Gtk.RadioMenuItem.new_with_label_from_widget(
                    model_group, size
                )
            model_radio.connect(
                'activate',
                lambda w, s=size: (
                    self.update_setting('model_size', s)
                    if w.get_active() else None
                )
            )
            model_radio.set_active(self.model_size == size)
            model_menu.append(model_radio)

        settings_menu.append(model_item)

        # 3. Language menu
        lang_item = Gtk.MenuItem(label="Language")
        lang_menu = Gtk.Menu()
        lang_item.set_submenu(lang_menu)

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
        lang_group = None
        for lang_name, lang_code in languages:
            if lang_group is None:
                lang_radio = Gtk.RadioMenuItem(label=lang_name)
                lang_group = lang_radio
            else:
                lang_radio = Gtk.RadioMenuItem.new_with_label_from_widget(
                    lang_group, lang_name
                )
            lang_radio.connect(
                'activate',
                lambda w, c=lang_code: (
                    self.update_setting('language', c)
                    if w.get_active() else None
                )
            )
            lang_radio.set_active(self.language == lang_code)
            lang_menu.append(lang_radio)

        settings_menu.append(lang_item)

        # 4. Speech Detection Settings submenu
        speech_item = Gtk.MenuItem(label="Speech Detection")
        speech_menu = Gtk.Menu()
        speech_item.set_submenu(speech_menu)

        # Min Speech Duration
        min_speech_item = Gtk.MenuItem(label="Min Speech Duration (s)")
        min_speech_menu = Gtk.Menu()
        min_speech_item.set_submenu(min_speech_menu)
        
        durations = [0.3, 0.5, 1.0, 1.5, 2.0]
        duration_group = None
        for duration in durations:
            if duration_group is None:
                duration_radio = Gtk.RadioMenuItem(label=str(duration))
                duration_group = duration_radio
            else:
                duration_radio = Gtk.RadioMenuItem.new_with_label_from_widget(
                    duration_group, str(duration)
                )
            duration_radio.connect(
                'activate',
                lambda w, d=duration: (
                    self.update_setting('min_speech_duration', d)
                    if w.get_active() else None
                )
            )
            duration_radio.set_active(
                abs(self.settings['min_speech_duration'] - duration) < 0.01
            )
            min_speech_menu.append(duration_radio)

        speech_menu.append(min_speech_item)

        # Max Silence Duration
        max_silence_item = Gtk.MenuItem(label="Max Silence Duration (s)")
        max_silence_menu = Gtk.Menu()
        max_silence_item.set_submenu(max_silence_menu)
        
        silences = [0.5, 1.0, 1.5, 2.0, 3.0]
        silence_group = None
        for silence in silences:
            if silence_group is None:
                silence_radio = Gtk.RadioMenuItem(label=str(silence))
                silence_group = silence_radio
            else:
                silence_radio = Gtk.RadioMenuItem.new_with_label_from_widget(
                    silence_group, str(silence)
                )
            silence_radio.connect(
                'activate',
                lambda w, s=silence: (
                    self.update_setting('max_silence_duration', s)
                    if w.get_active() else None
                )
            )
            silence_radio.set_active(
                abs(self.settings['max_silence_duration'] - silence) < 0.01
            )
            max_silence_menu.append(silence_radio)

        speech_menu.append(max_silence_item)

        # Speech Start Chunks
        chunks_item = Gtk.MenuItem(label="Speech Start Chunks")
        chunks_menu = Gtk.Menu()
        chunks_item.set_submenu(chunks_menu)
        
        for chunks in range(1, 6):
            if chunks == 1:
                chunks_radio = Gtk.RadioMenuItem(label=str(chunks))
                chunks_group = chunks_radio
            else:
                chunks_radio = Gtk.RadioMenuItem.new_with_label_from_widget(
                    chunks_group, str(chunks)
                )
            chunks_radio.connect(
                'activate',
                lambda w, c=chunks: (
                    self.update_setting('speech_start_chunks', c)
                    if w.get_active() else None
                )
            )
            chunks_radio.set_active(
                self.settings['speech_start_chunks'] == chunks
            )
            chunks_menu.append(chunks_radio)

        speech_menu.append(chunks_item)

        # Noise Reduction Threshold
        noise_item = Gtk.MenuItem(label="Noise Reduction")
        noise_menu = Gtk.Menu()
        noise_item.set_submenu(noise_menu)
        
        thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        threshold_group = None
        for threshold in thresholds:
            if threshold_group is None:
                threshold_radio = Gtk.RadioMenuItem(label=str(threshold))
                threshold_group = threshold_radio
            else:
                threshold_radio = Gtk.RadioMenuItem.new_with_label_from_widget(
                    threshold_group, str(threshold)
                )
            threshold_radio.connect(
                'activate',
                lambda w, t=threshold: (
                    self.update_setting('noise_reduce_threshold', t)
                    if w.get_active() else None
                )
            )
            threshold_radio.set_active(
                abs(self.settings['noise_reduce_threshold'] - threshold) < 0.01
            )
            noise_menu.append(threshold_radio)

        speech_menu.append(noise_item)

        settings_menu.append(speech_item)

        # 5. VAD sensitivity menu
        vad_item = Gtk.MenuItem(label="VAD Sensitivity")
        vad_menu = Gtk.Menu()
        vad_item.set_submenu(vad_menu)

        vad_group = None
        for level in range(4):
            if vad_group is None:
                vad_radio = Gtk.RadioMenuItem(label=str(level))
                vad_group = vad_radio
            else:
                vad_radio = Gtk.RadioMenuItem.new_with_label_from_widget(
                    vad_group, str(level)
                )
            vad_radio.connect(
                'activate',
                lambda w, l=level: (
                    self.update_setting('vad_sensitivity', l)
                    if w.get_active() else None
                )
            )
            vad_radio.set_active(self.vad_sensitivity == level)
            vad_menu.append(vad_radio)

        settings_menu.append(vad_item)

        # 6. Auto-detect speech toggle
        auto_detect = Gtk.CheckMenuItem(label="Auto-detect Speech")
        auto_detect.set_active(self.auto_detect_speech)
        auto_detect.connect(
            'toggled',
            lambda w: self.update_setting('auto_detect_speech', w.get_active())
        )
        settings_menu.append(auto_detect)

        # 7. Add punctuation toggle
        add_punct = Gtk.CheckMenuItem(label="Add Punctuation")
        add_punct.set_active(self.add_punctuation)
        add_punct.connect(
            'toggled',
            lambda w: self.update_setting('add_punctuation', w.get_active())
        )
        settings_menu.append(add_punct)

        self.menu.append(settings_item)

        # Quit item
        quit_item = Gtk.MenuItem(label="Quit")
        quit_item.connect('activate', self.quit)
        self.menu.append(quit_item)

        self.menu.show_all()

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

    def update_icon_state(self, state: str) -> None:
        """Update the tray icon state."""
        icon_names = {
            'ready': 'audio-input-microphone',
            'recording': 'audio-input-microphone-high',
            'computing': 'audio-input-microphone-medium',
            'error': 'audio-input-microphone-low',
            'no_mic': 'audio-input-microphone-muted'
        }
        self.indicator.set_icon(icon_names.get(state, 'audio-input-microphone'))

    def update_setting(self, key: str, value: Any) -> None:
        """Update a setting and save to config."""
        self.settings[key] = value
        save_settings(self.settings)
        
        # Apply setting changes
        if key == 'transcription_mode':
            if value == 'local':
                self.model = WhisperModel(
                    self.settings['model_size'], device="cpu", compute_type="int8"
                )
            elif value == 'openai':
                openai.api_key = self.settings['openai_api_key']
        
        elif key == 'model_size' and self.settings['transcription_mode'] == 'local':
            self.model = WhisperModel(
                value, device="cpu", compute_type="int8"
            )
        
        elif key == 'vad_sensitivity':
            self.vad = webrtcvad.Vad(value)
        
        elif key == 'use_clipboard':
            self.use_clipboard = value
            self.update_icon_state('ready')
        
        elif key == 'sample_rate':
            self.chunk_size = int(value * 0.03)
            # Restart audio thread with new sample rate
            if hasattr(self, 'audio_thread'):
                self.running = False
                self.audio_thread.join()
                self.running = True
                self.audio_thread = threading.Thread(
                    target=self.audio_loop, daemon=True
                )
                self.audio_thread.start()

    # ------------------------------
    def on_key_press(self, key: Optional[Union[Key, KeyCode]]) -> Optional[bool]:
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
        self.update_icon_state('recording')
        # Clear any previous audio data.
        self.audio_buffer = bytearray()

    def stop_recording(self) -> None:
        print("Manual recording stopped.")
        self.recording = False
        self.update_icon_state('ready')
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
                self.update_icon_state('error')
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
                    self.update_icon_state('recording')
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
                        self.update_icon_state('computing')
                        self.process_audio_buffer()
                        self.update_icon_state('ready')
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
                        self.update_icon_state('ready')

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
                self.update_icon_state('error')

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
                print(f"Error: Invalid transcription mode '{self.transcription_mode}'")
                return ""
                
        except Exception as e:
            print(f"Transcription error: {e}")
            self.update_icon_state('error')
            return ""

    # ------------------------------
    def quit(self, _: Any) -> None:
        """Clean up and quit the application."""
        self.running = False
        if hasattr(self, 'audio_thread'):
            self.audio_thread.join()
        Gtk.main_quit()

    def run(self) -> None:
        """Start the application."""
        # Start audio processing in a separate thread
        self.audio_thread = threading.Thread(target=self.audio_loop, daemon=True)
        self.audio_thread.start()
        Gtk.main()


# ------------------------------
if __name__ == "__main__":
    app = SpeechToTextApp()
    app.run() 