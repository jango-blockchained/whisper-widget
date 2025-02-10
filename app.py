import json
import os
import threading
import time
import wave

import gi
gi.require_version('Gtk', '3.0')
gi.require_version('AppIndicator3', '0.1')

import numpy as np
import openai  # for OpenAI transcription
import pyaudio
import pyperclip
import webrtcvad
from faster_whisper import WhisperModel
from gi.repository import Gtk, AppIndicator3
from pathlib import Path
from pynput.keyboard import Controller, Key, Listener


def check_microphone_access():
    """Check if we have access to the microphone."""
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
def noise_reduction(audio_data, sample_rate):
    # For now, simply return the data unmodified.
    return audio_data


def load_settings():
    """Load settings from config file."""
    config_dir = Path.home() / '.config' / 'whisper-widget'
    config_file = config_dir / 'config.json'
    default_settings = {
        'transcription_mode': 'continuous',  # or 'clipboard'
        'model_size': 'base',  # tiny, base, small, medium, large
        'language': 'en',  # Default to English instead of auto
        'vad_sensitivity': 3,  # 1-3
        'auto_detect_speech': True,
        'add_punctuation': True,
        'sample_rate': 16000,
        'silence_threshold': 400,
    }
    
    if config_file.exists():
        try:
            with open(config_file) as f:
                return {**default_settings, **json.load(f)}
        except Exception as e:
            print(f"Error loading settings: {e}")
            return default_settings
    return default_settings


def save_settings(settings):
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

    def __init__(self):
        """Initialize the application."""
        # Load settings
        self.settings = load_settings()
        
        # Initialize keyboard controller
        self.keyboard = Controller()
        
        # Initialize from settings
        self.transcription_mode = self.settings['transcription_mode']
        self.model_size = self.settings['model_size']
        self.language = self.settings['language']
        self.vad_sensitivity = self.settings['vad_sensitivity']
        self.auto_detect_speech = self.settings['auto_detect_speech']
        self.add_punctuation = self.settings['add_punctuation']
        self.sample_rate = self.settings['sample_rate']
        self.silence_threshold = self.settings['silence_threshold']
        
        # Initialize audio
        self.init_audio()
        
        # Initialize Whisper
        print("Initializing local faster-whisper model…")
        self.model = WhisperModel(
            self.model_size, device="cpu", compute_type="int8"
        )
        
        # Create system tray icon
        self.indicator = AppIndicator3.Indicator.new(
            'whisper-widget',
            'audio-input-microphone',
            AppIndicator3.IndicatorCategory.APPLICATION_STATUS
        )
        self.indicator.set_status(AppIndicator3.IndicatorStatus.ACTIVE)
        
        # Create menu
        self.menu = Gtk.Menu()
        
        # Add settings submenu
        settings_menu = Gtk.Menu()
        settings_item = Gtk.MenuItem.new_with_label('Settings')
        settings_item.set_submenu(settings_menu)
        
        # Add transcription mode submenu
        mode_menu = Gtk.Menu()
        mode_item = Gtk.MenuItem.new_with_label('Transcription Mode')
        mode_item.set_submenu(mode_menu)
        
        continuous_item = Gtk.RadioMenuItem.new_with_label(None, 'Continuous')
        continuous_item.connect(
            'activate',
            lambda _: self.update_setting('transcription_mode', 'continuous')
        )
        continuous_item.set_active(self.transcription_mode == 'continuous')
        mode_menu.append(continuous_item)
        
        clipboard_item = Gtk.RadioMenuItem.new_with_label_from_widget(
            continuous_item, 'Clipboard'
        )
        clipboard_item.connect(
            'activate',
            lambda _: self.update_setting('transcription_mode', 'clipboard')
        )
        clipboard_item.set_active(self.transcription_mode == 'clipboard')
        mode_menu.append(clipboard_item)
        
        settings_menu.append(mode_item)
        
        # Add model size submenu
        model_menu = Gtk.Menu()
        model_item = Gtk.MenuItem.new_with_label('Model Size')
        model_item.set_submenu(model_menu)
        
        model_sizes = ['tiny', 'base', 'small', 'medium', 'large']
        model_group = None
        for size in model_sizes:
            if model_group is None:
                model_group = Gtk.RadioMenuItem.new_with_label(
                    None, size.capitalize()
                )
            else:
                model_group = Gtk.RadioMenuItem.new_with_label_from_widget(
                    model_group, size.capitalize()
                )
            model_group.connect(
                'activate',
                lambda w, s=size: self.update_setting('model_size', s)
            )
            model_group.set_active(self.model_size == size)
            model_menu.append(model_group)
        
        settings_menu.append(model_item)
        
        # Add language submenu
        lang_menu = Gtk.Menu()
        lang_item = Gtk.MenuItem.new_with_label('Language')
        lang_item.set_submenu(lang_menu)
        
        languages = [
            ('English', 'en'),
            ('Spanish', 'es'),
            ('French', 'fr'),
            ('German', 'de'),
            ('Chinese', 'zh'),
            ('Japanese', 'ja'),
            ('Russian', 'ru')
        ]
        lang_group = None
        for name, code in languages:
            if lang_group is None:
                lang_group = Gtk.RadioMenuItem.new_with_label(None, name)
            else:
                lang_group = Gtk.RadioMenuItem.new_with_label_from_widget(
                    lang_group, name
                )
            lang_group.connect(
                'activate',
                lambda w, c=code: self.update_setting('language', c)
            )
            lang_group.set_active(self.language == code)
            lang_menu.append(lang_group)
        
        settings_menu.append(lang_item)
        
        # Add VAD sensitivity submenu
        vad_menu = Gtk.Menu()
        vad_item = Gtk.MenuItem.new_with_label('VAD Sensitivity')
        vad_item.set_submenu(vad_menu)
        
        vad_levels = [('Low', 1), ('Medium', 2), ('High', 3)]
        vad_group = None
        for name, level in vad_levels:
            if vad_group is None:
                vad_group = Gtk.RadioMenuItem.new_with_label(None, name)
            else:
                vad_group = Gtk.RadioMenuItem.new_with_label_from_widget(
                    vad_group, name
                )
            vad_group.connect(
                'activate',
                lambda w, level=level: self.update_setting('vad_sensitivity', level)
            )
            vad_group.set_active(self.vad_sensitivity == level)
            vad_menu.append(vad_group)
        
        settings_menu.append(vad_item)
        
        # Add toggle options
        auto_detect = Gtk.CheckMenuItem.new_with_label('Auto-detect Speech')
        auto_detect.set_active(self.auto_detect_speech)
        auto_detect.connect(
            'toggled',
            lambda w: self.update_setting('auto_detect_speech', w.get_active())
        )
        settings_menu.append(auto_detect)
        
        add_punct = Gtk.CheckMenuItem.new_with_label('Add Punctuation')
        add_punct.set_active(self.add_punctuation)
        add_punct.connect(
            'toggled',
            lambda w: self.update_setting('add_punctuation', w.get_active())
        )
        settings_menu.append(add_punct)
        
        self.menu.append(settings_item)
        
        # Add separator
        separator = Gtk.SeparatorMenuItem()
        self.menu.append(separator)
        
        # Add quit item
        quit_item = Gtk.MenuItem.new_with_label('Quit')
        quit_item.connect('activate', self.quit)
        self.menu.append(quit_item)
        
        self.menu.show_all()
        self.indicator.set_menu(self.menu)
        
        # Start background threads
        self.running = True
        self.recording = False
        self.audio_thread = threading.Thread(target=self.audio_loop)
        self.audio_thread.daemon = True
        self.audio_thread.start()

    def init_audio(self):
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
            self.chunk_size = int(self.sample_rate * 0.03)
            self.audio_format = pyaudio.paInt16
            self.channels = 1

            # Set up voice activity detection (VAD)
            self.vad = webrtcvad.Vad(self.vad_sensitivity)

    def update_icon_state(self, state):
        """Update the tray icon state."""
        icon_names = {
            'ready': 'audio-input-microphone',
            'recording': 'audio-input-microphone-high',
            'computing': 'audio-input-microphone-medium',
            'error': 'audio-input-microphone-low',
            'no_mic': 'audio-input-microphone-muted'
        }
        self.indicator.set_icon(icon_names.get(state, 'audio-input-microphone'))

    def update_setting(self, key, value):
        """Update a setting and save to config."""
        self.settings[key] = value
        save_settings(self.settings)
        
        # Apply setting changes
        if key == 'transcription_mode':
            if value == 'local':
                self.model = WhisperModel(
                    self.model_size, device="cpu", compute_type="int8"
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
            self.sample_rate = value
            # Restart audio thread with new sample rate
            if hasattr(self, 'audio_thread'):
                self.running = False
                self.audio_thread.join()
                self.chunk_size = int(self.sample_rate * 0.03)
                self.running = True
                self.audio_thread = threading.Thread(
                    target=self.audio_loop, daemon=True
                )
                self.audio_thread.start()

    # ------------------------------
    def on_key_press(self, key):
        """Toggle manual recording using the F9 key."""
        try:
            if key == Key.f9:
                if self.recording:
                    self.stop_recording()
                else:
                    self.start_recording()
        except Exception as e:
            print("Keyboard listener error:", e)

    # ------------------------------
    def start_recording(self, icon=None, item=None):
        if not self.has_mic_access:
            print("Cannot start recording: No microphone access")
            return

        print("Manual recording started.")
        self.recording = True
        self.update_icon_state('recording')
        # Clear any previous audio data.
        self.audio_buffer = bytearray()

    def stop_recording(self, icon=None, item=None):
        print("Manual recording stopped.")
        self.recording = False
        self.update_icon_state('ready')
        # If there is buffered audio, process it.
        if self.audio_buffer:
            self.process_audio_buffer()
            self.audio_buffer = bytearray()

    # ------------------------------
    def audio_loop(self):
        stream = self.p.open(
            format=self.audio_format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )
        # This threshold defines consecutive "silent" chunks for end-of-speech
        silent_chunks = 0

        while self.running:
            try:
                data = stream.read(
                    self.chunk_size, exception_on_overflow=False
                )
            except Exception as e:
                print("Audio stream error:", e)
                self.update_icon_state('error')
                continue

            # Determine if the chunk contains speech.
            is_speech = self.vad.is_speech(data, self.sample_rate)

            # If manually recording or if VAD detects speech, add to buffer.
            if self.recording or is_speech:
                if not self.recording:  # Auto-detection started
                    self.update_icon_state('recording')
                processed = noise_reduction(data, self.sample_rate)
                self.audio_buffer.extend(processed)
                silent_chunks = 0
            else:
                # If not recording and we have data, count silence.
                if self.audio_buffer:
                    silent_chunks += 1
                    if silent_chunks > self.silence_threshold:
                        print("Speech segment detected; processing…")
                        self.update_icon_state('computing')
                        self.process_audio_buffer()
                        self.audio_buffer = bytearray()
                        silent_chunks = 0
                        self.update_icon_state('ready')
            time.sleep(0.01)
        stream.stop_stream()
        stream.close()

    # ------------------------------
    def process_audio_buffer(self):
        # Save the buffered audio to a temporary WAV file.
        temp_filename = "temp_audio.wav"
        wf = wave.open(temp_filename, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.p.get_sample_size(self.audio_format))
        wf.setframerate(self.sample_rate)
        wf.writeframes(self.audio_buffer)
        wf.close()

        # Transcribe the temporary file.
        transcription = self.transcribe_audio(temp_filename)
        if transcription:
            self.output_text(transcription)

        # Clean up temporary file.
        os.remove(temp_filename)

    def output_text(self, text):
        """Output the text either to clipboard or type it directly."""
        if self.transcription_mode == 'clipboard':
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
    def transcribe_audio(self, audio_filename):
        try:
            if self.transcription_mode == "local":
                # Use the local faster-whisper model.
                segments, info = self.model.transcribe(
                    audio_filename,
                    beam_size=5,
                    word_timestamps=True,
                    language=self.language,
                    condition_on_previous_text=True,
                    no_speech_threshold=0.6
                )
                text = " ".join([seg.text for seg in segments])
                
                # Add basic punctuation if enabled
                if self.add_punctuation:
                    text = text.strip()
                    if text and not text[-1] in '.!?':
                        text += '.'
                
                return text
            elif self.transcription_mode == "openai":
                # Use OpenAI's API (ensure your API key is set).
                with open(audio_filename, "rb") as audio_file:
                    result = openai.Audio.transcribe("whisper-1", audio_file)
                    return result.get("text", "")
            else:
                return ""
        except Exception as e:
            print(f"Transcription error: {e}")
            self.update_icon_state('error')
            return ""

    # ------------------------------
    def quit(self, _):
        """Quit the application."""
        print("Quitting application…")
        self.running = False
        Gtk.main_quit()

    def run(self):
        """Run the application."""
        Gtk.main()


# ------------------------------
if __name__ == "__main__":
    app = SpeechToTextApp()
    app.run() 