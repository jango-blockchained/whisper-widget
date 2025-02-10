"""Speech-to-text application using GTK 4."""

from __future__ import annotations

import os
import threading
import time
import wave
from datetime import datetime
from typing import Any, Optional, Union, TypedDict
import unicodedata
import pathlib
import queue

import gi  # type: ignore
gi.require_version('Gtk', '4.0')
gi.require_version('WebKit', '6.0')
from gi.repository import (  # type: ignore # noqa: E402
    Gtk, Gio, GLib, WebKit, Gdk
)

import openai  # type: ignore # noqa: E402
import pyaudio  # type: ignore # noqa: E402
import pyperclip  # type: ignore # noqa: E402
import webrtcvad  # type: ignore # noqa: E402
import numpy as np  # type: ignore # noqa: E402
from faster_whisper import WhisperModel  # type: ignore # noqa: E402
from pynput.keyboard import (  # type: ignore # noqa: E402
    Key, KeyCode, Listener, Controller
)
import openwakeword  # type: ignore # noqa: E402


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
    line = (
        f"{current_time:12} | {status:10} | {duration_str:10} | "
        f"{mode:8} | {details}"
    )
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
        vad_sensitivity: int = 2,
        auto_detect_speech: bool = True,
        add_punctuation: bool = True,
        openai_api_key: Optional[str] = None,
        wake_word: str = "hey computer",
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
        print(f"Wake Word: {wake_word}")
        print("==========================================\n")
        
        print_status_header()
        
        # Initialize wake word detector
        self.wake_word = wake_word
        self.wake_word_detected = False
        self.wake_word_detector = openwakeword.Model()
        self.audio_buffer_ww = queue.Queue()
        
        # Start wake word detection thread
        self.wake_word_thread = threading.Thread(
            target=self._wake_word_loop,
            daemon=True
        )
        self.wake_word_thread.start()
        
        # Create application
        self.app = Gtk.Application.new(
            "com.github.whisper-widget",
            Gio.ApplicationFlags.FLAGS_NONE
        )
        self.app.connect("activate", self.on_activate)
        
        # Initialize window as None - will be created in on_activate
        self.window: Optional[Gtk.ApplicationWindow] = None
        
        # Initialize WebKit components
        self.webview: Optional[WebKit.WebView] = None
        
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
        self.min_speech_duration = 0.5
        self.max_silence_duration = 1.0
        self.min_audio_length = 0.5
        self.speech_threshold = 0.5
        self.silence_threshold = 15
        self.speech_start_chunks = 4
        self.noise_reduce_threshold = 0.15
        self.chunk_duration = 0.05
        self.chunk_size = int(self.sample_rate * self.chunk_duration)
        
        if openai_api_key:
            openai.api_key = openai_api_key
        
        # Initialize audio components
        self.init_audio()
        
        # Initialize Whisper model if using local transcription
        if self.transcription_mode == 'local':
            try:
                self.model = WhisperModel(
                    model_size_or_path=self.model_size,
                    device="cpu",
                    compute_type="int8"
                )
            except Exception as e:
                print(f"Error initializing Whisper model: {e}")
                self.model = None
        else:
            self.model = None

    def on_activate(self, app: Gtk.Application) -> None:
        """Handle application activation."""
        # Create the main window
        self.window = Gtk.ApplicationWindow.new(app)
        self.window.set_title("Whisper Widget")
        self.window.set_default_size(400, 300)
        
        # Make window transparent and borderless
        self.window.set_decorated(False)
        
        # Set up transparency
        css_provider = Gtk.CssProvider()
        css_provider.load_from_data(
            b"""
            window {
                background-color: transparent;
            }
            box {
                min-height: 100px;
            }
            """
        )
        Gtk.StyleContext.add_provider_for_display(
            Gdk.Display.get_default(),
            css_provider,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
        )
        
        # Create a box for layout
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        box.set_size_request(400, 100)  # Set minimum size
        self.window.set_child(box)
        
        # Create WebKit WebView
        self.webview = WebKit.WebView.new()
        
        # Set WebView background transparent
        self.webview.set_background_color(Gdk.RGBA())
        
        # Set WebView size
        self.webview.set_size_request(400, 100)
        
        # Load the visualization HTML file
        html_path = os.path.join(
            pathlib.Path(__file__).parent,
            'static',
            'index.html'
        )
        self.webview.load_uri(f'file://{html_path}')
        
        # Add WebView to box
        box.append(self.webview)
        
        # Create a popup menu
        menu = Gio.Menu.new()
        
        # Add transcription mode submenu
        trans_menu = Gio.Menu.new()
        for mode in ['local', 'openai']:
            action = f'app.trans_mode_{mode}'
            action_obj = Gio.SimpleAction.new_stateful(
                f'trans_mode_{mode}',
                None,
                GLib.Variant.new_boolean(self.transcription_mode == mode)
            )
            action_obj.connect('activate', self._on_trans_mode_change)
            self.app.add_action(action_obj)
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
        
        # Add Speech Detection Settings submenu
        detect_menu = Gio.Menu.new()
        
        # VAD Sensitivity submenu
        vad_menu = Gio.Menu.new()
        for level in [1, 2, 3]:
            action = f'app.vad_level_{level}'
            action_obj = Gio.SimpleAction.new_stateful(
                f'vad_level_{level}',
                None,
                GLib.Variant.new_boolean(self.vad_sensitivity == level)
            )
            action_obj.connect('activate', self._on_vad_level_change)
            self.app.add_action(action_obj)
            vad_menu.append(f'Level {level}', action)
        detect_menu.append_submenu('VAD Sensitivity', vad_menu)
        
        # Speech Start Settings
        start_menu = Gio.Menu.new()
        for chunks in [1, 2, 3, 4]:
            action = f'app.speech_start_{chunks}'
            action_obj = Gio.SimpleAction.new_stateful(
                f'speech_start_{chunks}',
                None,
                GLib.Variant.new_boolean(self.speech_start_chunks == chunks)
            )
            action_obj.connect('activate', self._on_speech_start_change)
            self.app.add_action(action_obj)
            start_menu.append(f'{chunks} chunk{"s" if chunks > 1 else ""}', action)
        detect_menu.append_submenu('Speech Start Threshold', start_menu)
        
        # Silence Settings
        silence_menu = Gio.Menu.new()
        for threshold in [3, 5, 7, 10]:
            action = f'app.silence_threshold_{threshold}'
            action_obj = Gio.SimpleAction.new_stateful(
                f'silence_threshold_{threshold}',
                None,
                GLib.Variant.new_boolean(self.silence_threshold == threshold)
            )
            action_obj.connect('activate', self._on_silence_threshold_change)
            self.app.add_action(action_obj)
            silence_menu.append(f'{threshold} chunks', action)
        detect_menu.append_submenu('Silence Threshold', silence_menu)
        
        # Duration Settings
        duration_menu = Gio.Menu.new()
        durations = [
            ('min_speech', 'Min Speech', [0.2, 0.5, 1.0], self.min_speech_duration),
            ('max_silence', 'Max Silence', [0.5, 1.0, 2.0], self.max_silence_duration),
            ('min_audio', 'Min Audio', [0.3, 0.5, 1.0], self.min_audio_length)
        ]
        for setting, label, values, current in durations:
            submenu = Gio.Menu.new()
            for value in values:
                action = f'app.{setting}_{str(value).replace(".", "_")}'
                action_obj = Gio.SimpleAction.new_stateful(
                    f'{setting}_{str(value).replace(".", "_")}',
                    None,
                    GLib.Variant.new_boolean(abs(current - value) < 0.01)
                )
                action_obj.connect('activate', self._on_duration_change)
                self.app.add_action(action_obj)
                submenu.append(f'{value}s', action)
            duration_menu.append_submenu(label, submenu)
        detect_menu.append_submenu('Duration Settings', duration_menu)
        
        menu.append_submenu('Speech Detection', detect_menu)
        
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
        
        # Add wake word detection toggle to the menu
        menu.append('Wake Word Detection', 'app.wake_word_toggle')
        self.app.add_action(
            Gio.SimpleAction.new_stateful(
                'wake_word_toggle',
                None,
                GLib.Variant.new_boolean(self.wake_word_detected)
            )
        )
        
        menu.append('Quit', 'app.quit')
        quit_action = Gio.SimpleAction.new('quit', None)
        quit_action.connect('activate', self.quit)
        self.app.add_action(quit_action)
        
        # Create a popover for the menu
        popover = Gtk.PopoverMenu.new_from_model(menu)
        
        # Make the window respond to right-click for menu
        right_click = Gtk.GestureClick.new()
        right_click.set_button(3)  # Right mouse button
        right_click.connect('pressed', self._on_right_click, popover)
        self.window.add_controller(right_click)
        
        # Make the window draggable
        drag = Gtk.GestureDrag.new()
        drag.connect('drag-begin', self._on_drag_begin)
        drag.connect('drag-update', self._on_drag_update)
        self.window.add_controller(drag)
        
        # Show the window
        self.window.present()

    def update_status(self, status: str) -> None:
        """Update the status and visualization."""
        if self.webview:
            # Map status to visualization state
            state = 'idle'
            if status == 'Recording':
                state = 'recording'
            elif status in ['Computing', 'Transcribing']:
                state = 'thinking'
            
            # Update visualization
            js = (
                'window.postMessage('
                f'{{"type": "state", "value": "{state}"}}, "*")'
            )
            self.webview.evaluate_javascript(js, -1, None, None, None)
        
        # Print status line
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

    def _on_trans_mode_change(
        self, 
        action: Gio.SimpleAction, 
        parameter: None
    ) -> None:
        """Handle transcription mode change."""
        mode = action.get_name().replace('trans_mode_', '')
        self.transcription_mode = mode
        
        # Update action state
        action.set_state(GLib.Variant.new_boolean(True))
        
        # Set other transcription mode actions to false
        for act in self.app.list_actions():
            is_trans_mode = act.startswith('trans_mode_')
            is_different_action = act != action.get_name()
            if is_trans_mode and is_different_action:
                self.app.lookup_action(act).set_state(
                    GLib.Variant.new_boolean(False)
                )
        
        # Update model availability
        if mode == 'local':
            try:
                self.model = WhisperModel(
                    model_size_or_path=self.model_size,
                    device="cpu",
                    compute_type="int8"
                )
            except Exception as e:
                error_msg = f"Error initializing Whisper model: {e}"
                print(error_msg)
                self.model = None
        else:  # openai
            if self.openai_api_key:
                openai.api_key = self.openai_api_key
            else:
                print("Warning: OpenAI API key not set")
                
        print_status_line(
            "Settings",
            mode=mode,
            details=f"Transcription mode changed to {mode}"
        )

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
            self.audio_format = pyaudio.paInt16
            self.channels = 1

            # Set up voice activity detection (VAD)
            try:
                self.vad = webrtcvad.Vad()
                self.vad.set_mode(min(max(self.vad_sensitivity, 0), 3))
            except Exception as e:
                print(f"Error initializing VAD: {e}")
                self.vad = None

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
            self.chunk_size = int(value * self.chunk_duration)
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
        
        # Reset wake word detection
        self.wake_word_detected = False

    # ------------------------------
    def audio_loop(self) -> None:
        """Process audio input in a loop."""
        if not self.has_mic_access or not self.vad:
            print("Cannot start audio loop - missing microphone access or VAD")
            return

        try:
            stream = self.p.open(
                format=self.audio_format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
        except Exception as e:
            print(f"Error opening audio stream: {e}")
            return

        while self.running:
            try:
                data = stream.read(self.chunk_size)
                if not data:
                    continue

                # Add audio data to wake word detection queue
                self.audio_buffer_ww.put(data)

                # Only process for speech if recording or auto-detect is on
                if self.is_recording or self.auto_detect_speech:
                    # Check for speech
                    try:
                        is_speech = self.vad.is_speech(
                            data,
                            sample_rate=self.sample_rate
                        )
                    except Exception as e:
                        print(f"VAD error: {e}")
                        continue

                    if is_speech:
                        self.audio_buffer.extend(data)
                    elif len(self.audio_buffer) > 0:
                        # Process the buffer if we have enough audio
                        if len(self.audio_buffer) >= self.min_audio_length * self.sample_rate:
                            self.process_audio()
                        self.audio_buffer = bytearray()
                        
                        # Reset wake word detection after processing
                        if self.wake_word_detected:
                            self.wake_word_detected = False

            except Exception as e:
                print(f"Error in audio loop: {e}")
                time.sleep(0.1)

        try:
            stream.stop_stream()
            stream.close()
        except Exception as e:
            print(f"Error closing audio stream: {e}")

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
                
                # Join all segment texts and handle encoding
                try:
                    # First try to normalize the text
                    text = " ".join([
                        unicodedata.normalize('NFKC', seg.text.strip())
                        for seg in segments
                    ])
                    # Then encode and decode with error handling
                    text = text.encode('utf-8', 'ignore').decode('utf-8')
                except Exception as e:
                    print(f"Warning: Text encoding issue: {e}")
                    # Fallback to a more aggressive normalization
                    try:
                        text = " ".join([
                            unicodedata.normalize('NFKD', seg.text.strip())
                            .encode('ascii', 'ignore')
                            .decode('ascii')
                            for seg in segments
                        ])
                    except Exception as e:
                        print(f"Error: Failed to normalize text: {e}")
                        return ""
                
                text = text.strip()
                
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

    def _on_vad_level_change(self, action: Gio.SimpleAction, parameter: None) -> None:
        """Handle VAD sensitivity level change."""
        level = int(action.get_name().replace('vad_level_', ''))
        self.vad_sensitivity = level
        
        if self.vad:
            try:
                self.vad.set_mode(min(max(level, 0), 3))
            except Exception as e:
                print(f"Error setting VAD mode: {e}")
        
        print_status_line(
            "Settings",
            details=f"VAD sensitivity changed to {level}"
        )
        
        # Update action state
        action.set_state(GLib.Variant.new_boolean(True))
        
        # Set other VAD level actions to false
        for act in self.app.list_actions():
            if act.startswith('vad_level_') and act != action.get_name():
                self.app.lookup_action(act).set_state(
                    GLib.Variant.new_boolean(False)
                )

    def _on_speech_start_change(self, action: Gio.SimpleAction, parameter: None) -> None:
        """Handle speech start threshold change."""
        chunks = int(action.get_name().replace('speech_start_', ''))
        self.speech_start_chunks = chunks
        
        print_status_line(
            "Settings",
            details=f"Speech start threshold: {chunks} chunks"
        )
        
        # Update action state
        action.set_state(GLib.Variant.new_boolean(True))
        
        # Set other speech start actions to false
        for act in self.app.list_actions():
            if act.startswith('speech_start_') and act != action.get_name():
                self.app.lookup_action(act).set_state(
                    GLib.Variant.new_boolean(False)
                )

    def _on_silence_threshold_change(self, action: Gio.SimpleAction, parameter: None) -> None:
        """Handle silence threshold change."""
        threshold = int(action.get_name().replace('silence_threshold_', ''))
        self.silence_threshold = threshold
        
        print_status_line(
            "Settings",
            details=f"Silence threshold: {threshold} chunks"
        )
        
        # Update action state
        action.set_state(GLib.Variant.new_boolean(True))
        
        # Set other silence threshold actions to false
        for act in self.app.list_actions():
            if act.startswith('silence_threshold_') and act != action.get_name():
                self.app.lookup_action(act).set_state(
                    GLib.Variant.new_boolean(False)
                )

    def _on_duration_change(self, action: Gio.SimpleAction, parameter: None) -> None:
        """Handle duration setting changes."""
        name = action.get_name()
        value = float(name.split('_')[-1].replace('_', '.'))
        setting = 'Unknown setting'  # Default value to prevent unassigned variable
        
        if name.startswith('min_speech_'):
            self.min_speech_duration = value
            setting = 'Min speech duration'
        elif name.startswith('max_silence_'):
            self.max_silence_duration = value
            setting = 'Max silence duration'
        elif name.startswith('min_audio_'):
            self.min_audio_length = value
            setting = 'Min audio length'
        
        print_status_line(
            "Settings",
            details=f"{setting} changed to {value}s"
        )
        
        # Update action state
        action.set_state(GLib.Variant.new_boolean(True))
        
        # Set other actions in the same group to false
        prefix = name[:name.rindex('_')]
        for act in self.app.list_actions():
            if act.startswith(prefix) and act != name:
                self.app.lookup_action(act).set_state(
                    GLib.Variant.new_boolean(False)
                )

    def _on_right_click(
        self,
        gesture: Gtk.GestureClick,
        n_press: int,
        x: float,
        y: float,
        popover: Gtk.PopoverMenu
    ) -> None:
        """Handle right-click to show menu."""
        popover.set_pointing_to(Gdk.Rectangle(x=x, y=y, width=1, height=1))
        popover.popup()

    def _on_drag_begin(
        self,
        gesture: Gtk.GestureDrag,
        start_x: float,
        start_y: float
    ) -> None:
        """Start window dragging."""
        # Store the initial window position
        self._drag_start_pos = self.window.get_position()

    def _on_drag_update(
        self,
        gesture: Gtk.GestureDrag,
        offset_x: float,
        offset_y: float
    ) -> None:
        """Update window position during drag."""
        if hasattr(self, '_drag_start_pos'):
            start_x, start_y = self._drag_start_pos
            self.window.move(
                int(start_x + offset_x),
                int(start_y + offset_y)
            )

    def _wake_word_loop(self) -> None:
        """Process audio for wake word detection."""
        while self.running:
            try:
                # Get audio data from the queue
                audio_chunk = self.audio_buffer_ww.get(timeout=0.1)
                
                # Convert to float32 for wake word detection
                audio_data = np.frombuffer(audio_chunk, dtype=np.int16)
                audio_data = audio_data.astype(np.float32) / 32768.0
                
                # Get wake word predictions
                predictions = self.wake_word_detector.predict(audio_data)
                
                # Check for wake word activation
                for prediction in predictions:
                    if prediction[1] > 0.5:  # Confidence threshold
                        if not self.wake_word_detected:
                            print_status_line(
                                "Wake Word",
                                details="Detected 'Hey Computer'"
                            )
                            self.wake_word_detected = True
                            self.start_recording()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Wake word detection error: {e}")
                time.sleep(0.1)

    def _on_wake_word_toggle(self, action: Gio.SimpleAction, parameter: None) -> None:
        """Handle wake word detection toggle."""
        state = not action.get_state()
        action.set_state(GLib.Variant.new_boolean(state))
        self.wake_word_detected = state
        print_status_line(
            "Settings",
            details=f"Wake word detection {'enabled' if state else 'disabled'}"
        )


def noise_reduction(
    audio_data: np.ndarray, 
    noise_threshold: float = 0.15
) -> np.ndarray:
    """
    Perform basic noise reduction on audio data.
    
    Args:
        audio_data (np.ndarray): Input audio data
        noise_threshold (float): Threshold for noise reduction
    
    Returns:
        np.ndarray: Noise-reduced audio data
    """
    # Simple noise reduction by zeroing out low-amplitude segments
    mask = np.abs(audio_data) > (noise_threshold * np.max(np.abs(audio_data)))
    return audio_data * mask.astype(audio_data.dtype)


# ------------------------------
if __name__ == "__main__":
    app = SpeechToTextApp()
    app.run() 