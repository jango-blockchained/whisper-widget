"""Tests for wake word detection functionality."""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from whisper_widget.app import SpeechToTextApp


@pytest.fixture
def mock_wake_word_detector():
    """Create a mock wake word detector."""
    with patch('openwakeword.Model') as mock_model:
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        mock_model_instance.predict.return_value = [(None, 0.0)]
        yield mock_model_instance


@pytest.fixture
def speech_app(mock_wake_word_detector):
    """Create a SpeechToTextApp instance for testing."""
    app = SpeechToTextApp(wake_word="hey computer")
    app.wake_word_detector = mock_wake_word_detector
    return app


def test_wake_word_initialization(speech_app):
    """Test that wake word detection is initialized correctly."""
    assert hasattr(speech_app, 'wake_word_detector')
    assert speech_app.wake_word == "hey computer"
    assert not speech_app.wake_word_detected


def test_wake_word_toggle(speech_app):
    """Test toggling wake word detection on and off."""
    # Initial state should be False
    assert not speech_app.wake_word_detected

    # Simulate toggling on
    speech_app.wake_word_detected = True
    assert speech_app.wake_word_detected

    # Simulate toggling off
    speech_app.wake_word_detected = False
    assert not speech_app.wake_word_detected


def test_wake_word_detection(speech_app, mock_wake_word_detector):
    """Test wake word detection mechanism."""
    # Create test audio data
    audio_data = np.random.randint(
        -32768, 32767,
        size=16000,
        dtype=np.int16
    ).tobytes()
    
    # Set up mock prediction
    mock_wake_word_detector.predict.return_value = [(None, 0.6)]
    
    # Add audio to buffer
    speech_app.audio_buffer_ww.put(audio_data)
    
    # Process one chunk
    speech_app._wake_word_loop()
    
    # Verify wake word was detected
    assert speech_app.wake_word_detected
    assert speech_app.is_recording


def test_wake_word_no_detection(speech_app, mock_wake_word_detector):
    """Test when wake word is not detected."""
    # Create test audio data
    audio_data = np.random.randint(
        -32768, 32767,
        size=16000,
        dtype=np.int16
    ).tobytes()
    
    # Set up mock prediction with low confidence
    mock_wake_word_detector.predict.return_value = [(None, 0.3)]
    
    # Add audio to buffer
    speech_app.audio_buffer_ww.put(audio_data)
    
    # Process one chunk
    speech_app._wake_word_loop()
    
    # Verify wake word was not detected
    assert not speech_app.wake_word_detected
    assert not speech_app.is_recording


def test_wake_word_error_handling(speech_app, mock_wake_word_detector):
    """Test error handling in wake word detection."""
    # Create test audio data
    audio_data = np.random.randint(
        -32768, 32767,
        size=16000,
        dtype=np.int16
    ).tobytes()
    
    # Set up mock to raise exception
    mock_wake_word_detector.predict.side_effect = Exception("Test error")
    
    # Add audio to buffer
    speech_app.audio_buffer_ww.put(audio_data)
    
    # Process one chunk - should not raise exception
    speech_app._wake_word_loop()
    
    # Verify state remains unchanged
    assert not speech_app.wake_word_detected
    assert not speech_app.is_recording 