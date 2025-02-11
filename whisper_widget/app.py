"""Speech-to-text application using GTK 3."""

from __future__ import annotations

import gi
gi.require_version('Gtk', '3.0')
gi.require_version('Gdk', '3.0')
gi.require_version('WebKit2', '4.1')
gi.require_version('AyatanaAppIndicator3', '0.1')

import logging
import os
import time
from datetime import datetime
from typing import Any, Optional

import pyperclip
from pynput.keyboard import Key, KeyCode, Listener as KeyboardListener
import numpy as np
import sys

# Import GTK components
from gi.repository import (
    Gtk,
    Gio,
    GLib,
    AyatanaAppIndicator3
)

from .audio.processor import AudioProcessor
from .audio.transcriber import Transcriber
from .ui.window import WhisperWindow
from .utils.audio_utils import check_microphone_access
from .ui.menu import create_app_menu

# Check if system tray is available
HAS_INDICATOR = False
try:
    gi.require_version('AyatanaAppIndicator3', '0.1')
    from gi.repository import AyatanaAppIndicator3
    HAS_INDICATOR = True
except (ValueError, ImportError):
    logging.warning("System tray support not available")

# ANSI color codes
COLORS = {
    'black': '\033[30m',
    'red': '\033[31m',
    'green': '\033[32m',
    'yellow': '\033[33m',
    'blue': '\033[34m',
    'magenta': '\033[35m',
    'cyan': '\033[36m',
    'white': '\033[37m',
    'bold': '\033[1m',
    'reset': '\033[0m'
}

# Status colors
STATUS_COLORS = {
    'IDLE': 'white',
    'LISTENING': 'cyan',
    'RECORDING': 'green',
    'TRANSCRIBING': 'yellow',
    'ERROR': 'red'
}

# ANSI control sequences
CONTROL = {
    'clear_screen': '\033[2J',
    'clear_line': '\033[K',
    'move_up': '\033[1A',
    'move_to_bottom': '\033[{};1H',
    'save_position': '\033[s',
    'restore_position': '\033[u',
    'hide_cursor': '\033[?25l',
    'show_cursor': '\033[?25h'
}

def color_text(text: str, color: str, bold: bool = False) -> str:
    """Add ANSI color codes to text."""
    color_code = COLORS.get(color, '')
    bold_code = COLORS['bold'] if bold else ''
    return f"{color_code}{bold_code}{text}{COLORS['reset']}"

def create_level_bar(level: float, width: int = 20) -> str:
    """Create a visual bar representation of the audio level."""
    if level <= 0:
        return "▁" * width
    
    # Map level to bar width (assuming level is between 0 and 1)
    filled = int(level * width)
    empty = width - filled
    
    # Use different block characters for a more granular visualization
    blocks = "▁▂▃▄▅▆▇█"
    bar = "█" * filled + "▁" * empty
    
    # Color the bar based on level
    if level > 0.8:
        return color_text(bar, 'red')
    elif level > 0.5:
        return color_text(bar, 'yellow')
    else:
        return color_text(bar, 'green')

def setup_cli_interface() -> None:
    """Set up the fixed CLI interface."""
    # Clear screen and hide cursor
    print(CONTROL['clear_screen'], end='')
    print(CONTROL['hide_cursor'], end='')
    # Print initial header
    print_status_header()

def cleanup_cli_interface() -> None:
    """Clean up the CLI interface."""
    print(CONTROL['show_cursor'], end='')

def print_status_header() -> None:
    """Print the status header with column names."""
    # Save cursor position
    print(CONTROL['save_position'], end='')
    # Move to bottom of terminal and clear lines
    print(CONTROL['move_to_bottom'].format(terminal_height() - 4), end='')
    print(CONTROL['clear_line'], end='')
    
    # Print header with better spacing
    print(color_text("=" * 100, 'blue', bold=True))
    print(color_text("Speech-to-Text Status Monitor", 'cyan', bold=True).center(100))
    print(color_text("-" * 100, 'blue', bold=True))
    header = (
        f"{color_text('Time', 'white', True):12} │ "
        f"{color_text('Status', 'white', True):12} │ "
        f"{color_text('Duration', 'white', True):10} │ "
        f"{color_text('Level', 'white', True):22} │ "
        f"{color_text('Mode', 'white', True):10} │ "
        f"{color_text('Details', 'white', True)}"
    )
    print(header)
    print(color_text("=" * 100, 'blue', bold=True))
    # Restore cursor position
    print(CONTROL['restore_position'], end='')

def print_status_line(
    status: str,
    duration: float = 0.0,
    mode: str = "",
    details: str = "",
    level: float = 0.0
) -> None:
    """Print a status line with current time and information."""
    # Save cursor position
    print(CONTROL['save_position'], end='')
    # Move to bottom of terminal
    print(CONTROL['move_to_bottom'].format(terminal_height()), end='')
    print(CONTROL['clear_line'], end='')
    
    current_time = datetime.now().strftime("%H:%M:%S")
    duration_str = f"{duration:.1f}s" if duration > 0 else ""
    
    # Create level visualization
    level_bar = create_level_bar(level)
    
    # Color the status
    status_color = STATUS_COLORS.get(status, 'white')
    colored_status = color_text(status.ljust(12), status_color, True)
    
    # Format the line with proper padding
    time_str = color_text(current_time.ljust(12), 'cyan')
    duration_str = color_text(duration_str.ljust(10), 'yellow')
    mode_str = color_text(mode.ljust(10), 'magenta')
    details_str = color_text(details, 'white')
    
    # Build and print the line with better separators
    line = (
        f"{time_str} │ {colored_status} │ {duration_str} │ "
        f"{level_bar} │ {mode_str} │ {details_str}"
    )
    print(line, end='')
    
    # Restore cursor position
    print(CONTROL['restore_position'], end='')
    sys.stdout.flush()

def terminal_height() -> int:
    """Get terminal height."""
    try:
        return os.get_terminal_size().lines
    except (AttributeError, OSError):
        return 24  # Fallback height

class SpeechToTextApp:
    """Main application class."""

    def __init__(
        self,
        transcription_mode: str = "local",
        output_mode: str = "continuous",
        model_size: str = "base",
        language: str = "de",
        vad_sensitivity: int = 2,
        auto_detect_speech: bool = True,
        add_punctuation: bool = True,
        openai_api_key: str | None = None,
        wake_word: str = "hey computer",
        use_tray: bool = True,
    ) -> None:
        """Initialize the application."""
        # Set up CLI interface
        setup_cli_interface()
        
        # Print initial header
        print(color_text("\nWhisper Widget - Speech-to-Text Application", 'cyan', True))
        print(color_text("=" * 42, 'blue'))
        print(f"{color_text('Transcription Mode:', 'white', True)} {color_text(transcription_mode, 'cyan')}")
        print(f"{color_text('Output Mode:', 'white', True)} {color_text(output_mode, 'cyan')}")
        print(f"{color_text('Model Size:', 'white', True)} {color_text(model_size, 'cyan')}")
        print(f"{color_text('Language:', 'white', True)} {color_text(language, 'cyan')}")
        print(f"{color_text('VAD Sensitivity:', 'white', True)} {color_text(str(vad_sensitivity), 'cyan')}")
        print(f"{color_text('Auto-detect Speech:', 'white', True)} {color_text(str(auto_detect_speech), 'cyan')}")
        print(f"{color_text('Add Punctuation:', 'white', True)} {color_text(str(add_punctuation), 'cyan')}")
        print(f"{color_text('Wake Word:', 'white', True)} {color_text(wake_word, 'cyan')}")
        print(color_text("=" * 42, 'blue'))
        print()
        
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
        self.wake_word = wake_word.lower()
        self.wake_word_detected = False
        self.use_tray = use_tray and HAS_INDICATOR
        
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
        
        # Initialize system tray indicator
        self.indicator = None
        if self.use_tray:
            self.setup_indicator()

    def setup_indicator(self) -> None:
        """Set up the system tray indicator."""
        self.indicator = AyatanaAppIndicator3.Indicator.new(
            "whisper-widget",
            "audio-input-microphone-symbolic",  # Use symbolic icon
            AyatanaAppIndicator3.IndicatorCategory.APPLICATION_STATUS
        )
        self.indicator.set_status(AyatanaAppIndicator3.IndicatorStatus.ACTIVE)
        
        # Create indicator menu from the application menu model
        menu = Gtk.Menu.new_from_model(create_app_menu(self.app))
        
        # Add recording toggle at the top
        toggle_item = Gtk.MenuItem(
            label="Start Recording" if not self.is_recording else "Stop Recording"
        )
        toggle_item.connect("activate", self._on_indicator_toggle)
        menu.prepend(toggle_item)
        menu.prepend(Gtk.SeparatorMenuItem())
        
        menu.show_all()
        self.indicator.set_menu(menu)

    def _on_indicator_toggle(self, _: Gtk.MenuItem) -> None:
        """Handle indicator menu toggle."""
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()

    def on_activate(self, app: Gtk.Application) -> None:
        """Handle application activation."""
        # Create main window
        self.window = WhisperWindow(
            app,
            on_recording_start=self.start_recording,
            on_recording_stop=self.stop_recording
        )
        
        # Set up window behavior based on tray availability
        if not self.use_tray:
            # If no tray icon, make window more visible but keep it borderless
            self.window.set_keep_above(True)
        
        self.window.show_all()

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
        
        # Update indicator menu if available
        if self.indicator:
            menu = self.indicator.get_menu()
            if menu:
                toggle_item = menu.get_children()[0]
                if isinstance(toggle_item, Gtk.MenuItem):
                    toggle_item.set_label("Stop Recording")

    def stop_recording(self) -> None:
        """Stop recording and reset recording state."""
        if self.is_recording:
            self.is_recording = False
            self.recording_start_time = None
            self.audio_processor.stop_recording()
            
            # Update indicator menu if available
            if self.indicator:
                menu = self.indicator.get_menu()
                if menu:
                    toggle_item = menu.get_children()[0]
                    if isinstance(toggle_item, Gtk.MenuItem):
                        toggle_item.set_label("Start Recording")

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
            
        # Get current audio level
        level = self.audio_processor.level if hasattr(self.audio_processor, 'level') else 0.0
            
        print_status_line(
            status,
            duration,
            self.transcriber.mode,
            ", ".join(details),
            level
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
                # Check for wake word if enabled
                text = text.lower().strip()
                if not self.wake_word_detected and self.wake_word in text:
                    self.wake_word_detected = True
                    print_status_line(
                        "Wake Word",
                        mode="detect",
                        details="Wake word detected"
                    )
                    return
                
                # Only process text if wake word is not required or has been detected
                if not self.wake_word or self.wake_word_detected:
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
                            print("Error typing text:", e)
                            print_status_line(
                                "Error",
                                mode="typing",
                                details=str(e)
                            )
            
            # Clean up temporary file
            try:
                os.remove(temp_filename)
            except Exception as e:
                logging.warning(f"Error removing temporary file: {e}")

    def quit(self, _: Any) -> None:
        """Quit the application."""
        if self.is_recording:
            self.stop_recording()
        if self.keyboard:
            self.keyboard.stop()
        cleanup_cli_interface()
        self.app.quit()

    def run(self) -> None:
        """Run the application."""
        self.app.run(None)


if __name__ == "__main__":
    app = SpeechToTextApp()
    app.run() 