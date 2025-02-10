"""Pytest configuration and shared fixtures."""

import os
import sys
from pathlib import Path
import tempfile
from typing import Generator, Dict, Any
from unittest.mock import MagicMock, patch

import pytest
import numpy as np
import pyaudio

# Ensure the project root is in the Python path
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')
))

# Mock GTK and related modules before they are imported
mock_gtk = MagicMock()
mock_gdk = MagicMock()
mock_webkit = MagicMock()
mock_gio = MagicMock()
mock_glib = MagicMock()
mock_app_indicator = MagicMock()

sys.modules['gi'] = MagicMock()
sys.modules['gi.repository'] = MagicMock()
sys.modules['gi.repository.Gtk'] = mock_gtk
sys.modules['gi.repository.Gdk'] = mock_gdk
sys.modules['gi.repository.WebKit2'] = mock_webkit
sys.modules['gi.repository.Gio'] = mock_gio
sys.modules['gi.repository.GLib'] = mock_glib
sys.modules['gi.repository.AppIndicator3'] = mock_app_indicator

# Set up common mock objects
mock_gtk.ApplicationWindow = MagicMock
mock_gtk.Application = MagicMock
mock_gtk.Box = MagicMock
mock_gtk.CssProvider = MagicMock
mock_gtk.StyleContext = MagicMock
mock_gtk.GestureClick = MagicMock
mock_gtk.GestureDrag = MagicMock
mock_gtk.PopoverMenu = MagicMock
mock_gtk.Orientation = MagicMock
mock_gtk.STYLE_PROVIDER_PRIORITY_APPLICATION = 1

mock_gdk.Display = MagicMock
mock_gdk.Rectangle = MagicMock
mock_gdk.RGBA = MagicMock

mock_webkit.WebView = MagicMock

mock_gio.Menu = MagicMock
mock_gio.SimpleAction = MagicMock
mock_gio.ApplicationFlags = MagicMock
mock_gio.ApplicationFlags.FLAGS_NONE = 0

mock_glib.Variant = MagicMock
mock_glib.Variant.new_boolean = lambda x: x

mock_app_indicator.Indicator = MagicMock
mock_app_indicator.IndicatorCategory = MagicMock
mock_app_indicator.IndicatorCategory.APPLICATION_STATUS = 1

class MockStream:
    """Mock audio stream for testing."""
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.is_active: bool = True
        self._frames: list = []
    
    def start_stream(self) -> None:
        """Start the mock stream."""
        self.is_active = True
    
    def stop_stream(self) -> None:
        """Stop the mock stream."""
        self.is_active = False
    
    def close(self) -> None:
        """Close the mock stream."""
        self.is_active = False
    
    def read(
        self, 
        chunk_size: int, 
        exception_on_overflow: bool = True
    ) -> bytes:
        """
        Read mock audio data.
        
        Args:
            chunk_size (int): Size of audio chunk
            exception_on_overflow (bool): Whether to raise on overflow
        
        Returns:
            bytes: Mock audio data
        """
        if not self.is_active:
            return b''
        return np.random.bytes(chunk_size * 2)


class MockPyAudio:
    """Mock PyAudio class for testing."""
    def __init__(self) -> None:
        self.format: int = pyaudio.paInt16
        self.streams: list[MockStream] = []
    
    def get_default_input_device_info(self) -> Dict[str, Any]:
        """Return mock input device info."""
        return {'name': 'Mock Input Device'}

    def open(self, *args: Any, **kwargs: Any) -> MockStream:
        """Return mock stream."""
        stream = MockStream()
        self.streams.append(stream)
        return stream
    
    def get_device_count(self) -> int:
        """Get number of mock devices."""
        return 1
    
    def get_sample_size(self, format_type: int) -> int:
        """Return mock sample size."""
        return 2  # 16-bit audio = 2 bytes
    
    def terminate(self) -> None:
        """Mock terminate method."""
        for stream in self.streams:
            stream.close()


@pytest.fixture
def mock_audio() -> Generator[None, None, None]:
    """Mock PyAudio for testing."""
    with patch('pyaudio.PyAudio', return_value=MockPyAudio()):
        yield


@pytest.fixture
def mock_gtk() -> Generator[MagicMock, None, None]:
    """Mock GTK and AppIndicator3."""
    with patch('gi.repository.Gtk') as mock_gtk, \
         patch('gi.repository.AppIndicator3') as mock_indicator:
        # Create mock menu items
        mock_menu = MagicMock()
        mock_menu_item = MagicMock()
        mock_submenu = MagicMock()
        
        # Set up menu hierarchy
        mock_menu_item.get_submenu.return_value = mock_submenu
        mock_menu.get_children.return_value = [mock_menu_item]
        mock_menu.get_parent.return_value = mock_gtk
        
        # Set up GTK mocks
        mock_gtk.Menu.return_value = mock_menu
        mock_gtk.MenuItem = MagicMock
        mock_gtk.RadioMenuItem = MagicMock
        mock_gtk.CheckMenuItem = MagicMock
        mock_gtk.SeparatorMenuItem = MagicMock
        
        # Set up AppIndicator mocks
        mock_indicator.Indicator.new.return_value = MagicMock()
        mock_indicator.IndicatorCategory = MagicMock()
        mock_indicator.IndicatorStatus = MagicMock()
        
        yield mock_gtk


@pytest.fixture
def mock_app(
    mock_audio: None, 
    mock_gtk: MagicMock, 
    temp_config_dir: Path
) -> Any:
    """Create a mock app with all components mocked."""
    with patch('whisper_widget.app.WhisperModel') as mock_model, \
         patch('whisper_widget.app.webrtcvad.Vad') as mock_vad:
        
        # Set up mock model
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        
        # Set up mock VAD
        mock_vad_instance = MagicMock()
        mock_vad.return_value = mock_vad_instance
        mock_vad_instance.is_speech.return_value = True
        
        # Import here to avoid circular imports
        from whisper_widget.app import SpeechToTextApp
        app = SpeechToTextApp()
        
        # Replace real components with mocks
        app.model = mock_model_instance
        app.vad = mock_vad_instance
        app.indicator = MagicMock()
        
        return app


@pytest.fixture
def temp_config_dir() -> Generator[Path, None, None]:
    """Create a temporary config directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        old_home = os.environ.get('HOME')
        os.environ['HOME'] = temp_dir
        yield Path(temp_dir)
        if old_home:
            os.environ['HOME'] = old_home


@pytest.fixture
def sample_settings() -> Dict[str, Any]:
    """Return sample settings for testing."""
    return {
        'transcription_mode': 'continuous',
        'model_size': 'base',
        'language': 'en',
        'vad_sensitivity': 3,
        'auto_detect_speech': True,
        'add_punctuation': True,
        'sample_rate': 16000,
        'silence_threshold': 400,
    }


@pytest.fixture
def sample_audio_file() -> Generator[str, None, None]:
    """Create a sample WAV file for testing."""
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        # Create a simple WAV file with 1 second of silence
        import wave
        import struct
        
        frames_per_second = 16000
        num_frames = frames_per_second
        
        with wave.open(temp_file.name, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(frames_per_second)
            for _ in range(num_frames):
                wf.writeframes(struct.pack('h', 0))
        
        yield temp_file.name
        os.unlink(temp_file.name)


@pytest.fixture
def mock_wake_word_detector(mocker: Any) -> MagicMock:
    """Fixture to mock the wake word detector."""
    from openwakeword import Model
    mock_model = mocker.Mock(spec=Model)
    mock_model.predict.return_value = [(None, 0.0)]
    return mock_model


@pytest.fixture
def default_wake_word() -> str:
    """Fixture providing a default wake word."""
    return "hey computer" 