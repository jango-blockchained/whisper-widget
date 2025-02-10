"""Tests for audio transcription functionality."""

import pytest
from unittest.mock import MagicMock, patch
import wave
import numpy as np
import openai

from whisper_widget.audio.transcriber import Transcriber


@pytest.fixture
def mock_transcriber():
    """Create a mock transcriber for testing."""
    with patch('whisper_widget.audio.transcriber.WhisperModel') as mock_model:
        # Set up mock model
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        
        transcriber = Transcriber()
        
        # Replace real model with mock
        transcriber.model = mock_model_instance
        
        return transcriber


def test_transcriber_initialization():
    """Test Transcriber initialization with different modes."""
    # Test local mode
    with patch('whisper_widget.audio.transcriber.WhisperModel') as mock_model:
        transcriber = Transcriber(mode="local")
        assert transcriber.mode == "local"
        assert mock_model.called
    
    # Test OpenAI mode
    transcriber = Transcriber(mode="openai", openai_api_key="test_key")
    assert transcriber.mode == "openai"
    assert openai.api_key == "test_key"


def test_local_transcription(mock_transcriber, tmp_path):
    """Test local transcription with Whisper model."""
    # Create a test WAV file
    test_file = tmp_path / "test.wav"
    with wave.open(str(test_file), 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(np.random.bytes(16000 * 2))
    
    # Set up mock transcription result
    mock_segment = MagicMock()
    mock_segment.text = "Test transcription"
    mock_transcriber.model.transcribe.return_value = ([mock_segment], None)
    
    # Test transcription
    result = mock_transcriber.transcribe(str(test_file))
    assert result == "Test transcription."  # Note: adds period due to add_punctuation
    
    # Verify model was called with correct parameters
    mock_transcriber.model.transcribe.assert_called_with(
        str(test_file),
        beam_size=5,
        word_timestamps=True,
        language=mock_transcriber.language,
        condition_on_previous_text=True,
        no_speech_threshold=0.6
    )


def test_openai_transcription(tmp_path):
    """Test OpenAI API transcription."""
    # Create a test WAV file
    test_file = tmp_path / "test.wav"
    with wave.open(str(test_file), 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(np.random.bytes(16000 * 2))
    
    # Create transcriber in OpenAI mode
    transcriber = Transcriber(
        mode="openai",
        openai_api_key="test_key"
    )
    
    # Mock OpenAI API call
    with patch('openai.Audio.transcribe') as mock_transcribe:
        mock_transcribe.return_value = {"text": "OpenAI transcription"}
        
        result = transcriber.transcribe(str(test_file))
        assert result == "OpenAI transcription"
        
        # Verify API was called correctly
        mock_transcribe.assert_called_once()
        assert mock_transcribe.call_args[0][0] == "whisper-1"


def test_transcription_error_handling(mock_transcriber, tmp_path):
    """Test error handling during transcription."""
    # Create a test WAV file
    test_file = tmp_path / "test.wav"
    with wave.open(str(test_file), 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(np.random.bytes(16000 * 2))
    
    # Test local transcription error
    mock_transcriber.model.transcribe.side_effect = Exception("Test error")
    result = mock_transcriber.transcribe(str(test_file))
    assert result == ""
    
    # Test OpenAI transcription error
    transcriber = Transcriber(
        mode="openai",
        openai_api_key="test_key"
    )
    with patch('openai.Audio.transcribe', side_effect=Exception("API error")):
        result = transcriber.transcribe(str(test_file))
        assert result == ""


def test_transcriber_settings_update(mock_transcriber):
    """Test updating transcriber settings."""
    # Test mode change
    mock_transcriber.update_settings(mode="openai", openai_api_key="new_key")
    assert mock_transcriber.mode == "openai"
    assert openai.api_key == "new_key"
    
    # Test model size change
    mock_transcriber.update_settings(mode="local", model_size="small")
    assert mock_transcriber.model_size == "small"
    assert mock_transcriber.model is not None
    
    # Test language change
    mock_transcriber.update_settings(language="fr")
    assert mock_transcriber.language == "fr"
    
    # Test punctuation setting
    mock_transcriber.update_settings(add_punctuation=False)
    assert not mock_transcriber.add_punctuation


def test_text_normalization(mock_transcriber, tmp_path):
    """Test text normalization in transcription."""
    # Create a test WAV file
    test_file = tmp_path / "test.wav"
    with wave.open(str(test_file), 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(np.random.bytes(16000 * 2))
    
    # Test normal text
    mock_segment = MagicMock()
    mock_segment.text = "Test transcription"
    mock_transcriber.model.transcribe.return_value = ([mock_segment], None)
    
    result = mock_transcriber.transcribe(str(test_file))
    assert result == "Test transcription."
    
    # Test text with special characters
    mock_segment.text = "Test with é and ñ"
    result = mock_transcriber.transcribe(str(test_file))
    assert "Test with" in result
    assert "and" in result
    
    # Test text without punctuation
    mock_transcriber.add_punctuation = False
    result = mock_transcriber.transcribe(str(test_file))
    assert not result.endswith(".") 