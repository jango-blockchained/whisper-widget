import os
import pytest
from pathlib import Path
import tempfile
import json
from unittest.mock import MagicMock, patch
import numpy as np
import pyaudio


class MockStream:
    def __init__(self, *args, **kwargs):
        self.is_active = True
        self._frames = []
    
    def start_stream(self):
        self.is_active = True
    
    def stop_stream(self):
        self.is_active = False
    
    def close(self):
        self.is_active = False
    
    def read(self, chunk_size, exception_on_overflow=True):
        if not self.is_active:
            return None
        return np.random.bytes(chunk_size * 2)


class MockPyAudio:
    """Mock PyAudio class for testing."""
    def __init__(self):
        self.format = pyaudio.paInt16
        self.streams = []
    
    def get_default_input_device_info(self):
        """Return mock input device info."""
        return {'name': 'Mock Input Device'}

    def open(self, *args, **kwargs):
        """Return mock stream."""
        stream = MockStream()
        self.streams.append(stream)
        return stream
    
    def get_device_count(self):
        return 1
    
    def get_sample_size(self, format_type):
        """Return mock sample size."""
        return 2  # 16-bit audio = 2 bytes
    
    def terminate(self):
        """Mock terminate method."""
        for stream in self.streams:
            stream.close()


@pytest.fixture
def mock_audio():
    """Mock PyAudio for testing."""
    with patch('pyaudio.PyAudio', return_value=MockPyAudio()):
        yield


@pytest.fixture
def mock_gtk():
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
def mock_app(mock_audio, mock_gtk, temp_config_dir):
    """Create a mock app with all components mocked."""
    with patch('app.WhisperModel') as mock_model, \
         patch('app.webrtcvad.Vad') as mock_vad:
        
        # Set up mock model
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        
        # Set up mock VAD
        mock_vad_instance = MagicMock()
        mock_vad.return_value = mock_vad_instance
        mock_vad_instance.is_speech.return_value = True
        
        # Import here to avoid circular imports
        from app import SpeechToTextApp
        app = SpeechToTextApp()
        
        # Replace real components with mocks
        app.model = mock_model_instance
        app.vad = mock_vad_instance
        app.indicator = MagicMock()
        
        return app


@pytest.fixture
def temp_config_dir():
    """Create a temporary config directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        old_home = os.environ.get('HOME')
        os.environ['HOME'] = temp_dir
        yield Path(temp_dir)
        if old_home:
            os.environ['HOME'] = old_home


@pytest.fixture
def sample_settings():
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
def sample_audio_file():
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