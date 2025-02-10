import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from whisper_widget import (
    check_microphone_access,
    noise_reduction,
    SpeechToTextApp
)


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
        return np.random.bytes(chunk_size * 2)  # 16-bit audio = 2 bytes per sample


class MockPyAudio:
    def __init__(self):
        self.streams = []
    
    def open(self, format=None, channels=None, rate=None, input=None,
             frames_per_buffer=None, stream_callback=None, start=True,
             input_device_index=None, **kwargs):
        stream = MockStream()
        self.streams.append(stream)
        return stream
    
    def get_default_input_device_info(self):
        return {'name': 'Mock Input Device', 'index': 0}
    
    def get_device_count(self):
        return 1
    
    def terminate(self):
        for stream in self.streams:
            stream.close()


@pytest.fixture
def mock_audio():
    with patch('pyaudio.PyAudio', return_value=MockPyAudio()):
        yield


def test_check_microphone_access(mock_audio):
    """Test microphone access check with mock audio."""
    assert check_microphone_access() is True


def test_check_microphone_access_no_device():
    """Test microphone access check with no device available."""
    with patch('pyaudio.PyAudio') as mock_pyaudio:
        mock_instance = mock_pyaudio.return_value
        mock_instance.get_default_input_device_info.side_effect = OSError("No device found")
        assert check_microphone_access() is False


def test_noise_reduction():
    """Test noise reduction function."""
    # Create sample audio data
    audio_data = b'test_audio_data'
    sample_rate = 16000
    
    # Apply noise reduction
    reduced = noise_reduction(audio_data, sample_rate)
    
    # Check that output is same as input (since it's a passthrough)
    assert reduced == audio_data


@pytest.fixture
def mock_whisper_app(mock_audio):
    """Create a mock whisper app with mocked components."""
    with patch('whisper_widget.WhisperModel') as mock_model, \
         patch('whisper_widget.webrtcvad.Vad') as mock_vad:
        
        # Set up mock model
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        
        # Set up mock VAD
        mock_vad_instance = MagicMock()
        mock_vad.return_value = mock_vad_instance
        
        app = SpeechToTextApp()
        
        # Replace the real model with our mock
        app.model = mock_model_instance
        app.vad = mock_vad_instance
        
        return app


def test_app_initialization(mock_whisper_app):
    """Test app initialization."""
    app = mock_whisper_app
    
    # Check that components are initialized
    assert app.model is not None
    assert app.vad is not None
    assert app.indicator is not None
    assert app.menu is not None
    
    # Check default settings
    assert app.transcription_mode in ['continuous', 'clipboard']
    assert app.model_size in ['tiny', 'base', 'small', 'medium', 'large']
    assert app.language in ['en', 'es', 'fr', 'de', 'zh', 'ja', 'ru']
    assert 1 <= app.vad_sensitivity <= 3 