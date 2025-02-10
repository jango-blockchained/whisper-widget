"""Speech-to-text application using GTK 4."""

from __future__ import annotations

import os
import threading
import time
from datetime import datetime
import logging
import pyperclip
import gi

gi.require_version('Gtk', '4.0')
gi.require_version('AppIndicator3', '0.1')

from gi.repository import Gtk, AppIndicator3
from pynput.keyboard import Key, KeyCode, Listener as KeyboardListener

from .audio.processor import AudioProcessor
from .audio.transcriber import Transcriber
from .ui.window import WhisperWindow
from .utils.audio_utils import check_microphone_access


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
        openai_api_key: str | None = None,
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
        
        # Initialize components
        self.transcriber = Transcriber(
            mode=transcription_mode,
            model_size=model_size,
            language=language,
            add_punctuation=add_punctuation,
            openai_api_key=openai_api_key
        )
        
        self.audio_processor = AudioProcessor(
            vad_sensitivity=vad_sensitivity,
            on_speech_detected=self._on_speech_detected
        )
        
        # Store settings
        self.output_mode = output_mode
        self.auto_detect_speech = auto_detect_speech
        self.wake_word = wake_word
        self.wake_word_detected = False
        
        # Initialize recording state
        self.is_recording = False
        self.recording_start_time = None
        
        # Create application
        self.app = Gtk.Application.new(
            "com.github.whisper-widget",
            Gio.ApplicationFlags.FLAGS_NONE
        )
        self.app.connect("activate", self.on_activate)
        
        # Initialize keyboard listener
        self.keyboard = KeyboardListener(on_press=self.on_key_press)
        self.keyboard.start()
        
        # Initialize AppIndicator
        if AppIndicator3:
            self.indicator = AppIndicator3.Indicator.new(
                "whisper-widget",
                "audio-input-microphone",
                AppIndicator3.IndicatorCategory.APPLICATION_STATUS
            )
        else:
            self.indicator = None

    def on_activate(self, app: Gtk.Application) -> None:
        """Handle application activation."""
        # Create main window
        self.window = WhisperWindow(
            app,
            on_recording_start=self.start_recording,
            on_recording_stop=self.stop_recording
        )
        self.window.present()

    def on_key_press(
        self,
        key: Key | KeyCode | None
    ) -> bool:
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

    def start_recording(self) -> None:
        """Start recording audio."""
        if not check_microphone_access():
            print_status_line("Error", details="No microphone access")
            self.update_status('No Mic')
            return

        self.recording_start_time = time.time()
        self.is_recording = True
        self.audio_processor.start_recording()
        self.update_status('Recording')

    def stop_recording(self) -> None:
        """Stop recording and reset recording state."""
        if self.is_recording:
            self.is_recording = False
            self.recording_start_time = None
            self.audio_processor.stop_recording()

    def update_status(self, status: str) -> None:
        """Update the status and visualization."""
        if self.window:
            # Map status to visualization state
            state = 'idle'
            if status == 'Recording':
                state = 'recording'
            elif status in ['Computing', 'Transcribing']:
                state = 'thinking'
            
            # Update visualization
            self.window.update_visualization(state)
        
        # Print status line
        duration = 0.0
        if self.recording_start_time and self.is_recording:
            duration = time.time() - self.recording_start_time
        
        details = []
        if self.transcriber.mode == "local":
            details.append(f"Model: {self.transcriber.model_size}")
        if self.auto_detect_speech:
            details.append("Auto-detect")
        if self.transcriber.add_punctuation:
            details.append("Punctuation")
            
        print_status_line(
            status,
            duration,
            self.transcriber.mode,
            ", ".join(details)
        )

    def _on_speech_detected(self, audio_data: bytes) -> None:
        """Handle detected speech segment."""
        # Save to temporary file
        temp_filename = "temp_audio.wav"
        if self.audio_processor.save_audio(audio_data, temp_filename):
            # Transcribe
            self.update_status('Transcribing')
            text = self.transcriber.transcribe(temp_filename)
            
            if text:
                # Output text
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
            
            # Clean up
            try:
                os.remove(temp_filename)
            except Exception as e:
                logging.warning(f"Error removing temporary file: {e}")

    def quit(self, _: Any) -> None:
        """Clean up and quit the application."""
        self.is_recording = False
        self.audio_processor.cleanup()
        self.keyboard.stop()
        self.app.quit()

    def run(self) -> None:
        """Start the application."""
        self.app.run(None)


if __name__ == "__main__":
    app = SpeechToTextApp()
    app.run() 