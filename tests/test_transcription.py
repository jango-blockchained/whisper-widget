import pytest
from unittest.mock import MagicMock, patch
import numpy as np
import wave
from app import SpeechToTextApp
import sys


@pytest.fixture
def mock_app(mock_audio):
    """Create a mock app for transcription testing."""
    with patch('app.WhisperModel') as mock_model, \
         patch('app.webrtcvad.Vad') as mock_vad:
        
        # Set up mock model
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        
        # Set up mock VAD
        mock_vad_instance = MagicMock()
        mock_vad.return_value = mock_vad_instance
        
        app = SpeechToTextApp()
        
        # Replace real components with mocks
        app.model = mock_model_instance
        app.vad = mock_vad_instance
        app.indicator = MagicMock()
        
        return app


def test_local_transcription(mock_app, tmp_path):
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
    mock_app.model.transcribe.return_value = ([mock_segment], None)
    
    # Ensure we're in local mode and disable punctuation
    mock_app.transcription_mode = "local"
    mock_app.add_punctuation = False
    
    # Test transcription
    result = mock_app.transcribe_audio(str(test_file))
    assert result == "Test transcription"
    
    # Verify model was called with correct parameters
    mock_app.model.transcribe.assert_called_with(
        str(test_file),
        beam_size=5,
        word_timestamps=True,
        language=mock_app.language,
        condition_on_previous_text=True,
        no_speech_threshold=0.6
    )


def test_openai_transcription(mock_app, tmp_path):
    """Test OpenAI API transcription."""
    # Create a test WAV file
    test_file = tmp_path / "test.wav"
    with wave.open(str(test_file), 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(np.random.bytes(16000 * 2))
    
    # Set up OpenAI mode
    mock_app.transcription_mode = "openai"
    mock_app.settings['openai_api_key'] = "test_key"
    
    # Create a mock OpenAI module
    mock_openai = MagicMock()
    mock_openai.Audio.transcribe.return_value = {"text": "OpenAI transcription"}
    
    # Patch the openai module at the function level
    with patch('app.openai', mock_openai):
        result = mock_app.transcribe_audio(str(test_file))
        assert result == "OpenAI transcription"
        mock_openai.Audio.transcribe.assert_called_once_with(
            "whisper-1", mock_openai.Audio.transcribe.call_args[0][1]
        )


def test_transcription_error_handling(mock_app, tmp_path):
    """Test error handling during transcription."""
    # Create a test WAV file
    test_file = tmp_path / "test.wav"
    with wave.open(str(test_file), 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(np.random.bytes(16000 * 2))
    
    # Ensure we're in local mode
    mock_app.transcription_mode = "local"
    
    # Test local transcription error
    mock_app.model.transcribe.side_effect = Exception("Test error")
    result = mock_app.transcribe_audio(str(test_file))
    assert result == ""
    mock_app.indicator.set_icon.assert_called()
    
    # Test OpenAI transcription error
    mock_app.transcription_mode = "openai"
    
    # Create a mock OpenAI module that raises an error
    mock_openai = MagicMock()
    mock_openai.Audio.transcribe.side_effect = Exception("API error")
    
    # Patch the openai module at the function level
    with patch('app.openai', mock_openai):
        result = mock_app.transcribe_audio(str(test_file))
        assert result == ""
        mock_app.indicator.set_icon.assert_called()


def test_transcription_with_punctuation(mock_app, tmp_path):
    """Test transcription with punctuation enabled."""
    # Create a test WAV file
    test_file = tmp_path / "test.wav"
    with wave.open(str(test_file), 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(np.random.bytes(16000 * 2))
    
    # Enable punctuation and ensure local mode
    mock_app.add_punctuation = True
    mock_app.transcription_mode = "local"
    
    # Test with text not ending in punctuation
    mock_segment = MagicMock()
    mock_segment.text = "Test transcription"
    mock_app.model.transcribe.return_value = ([mock_segment], None)
    
    result = mock_app.transcribe_audio(str(test_file))
    assert result == "Test transcription."
    
    # Test with text already ending in punctuation
    mock_segment.text = "Test transcription!"
    mock_app.model.transcribe.return_value = ([mock_segment], None)
    
    result = mock_app.transcribe_audio(str(test_file))
    assert result == "Test transcription!"


def test_invalid_transcription_mode(mock_app, tmp_path):
    """Test transcription with invalid mode."""
    # Create a test WAV file
    test_file = tmp_path / "test.wav"
    with wave.open(str(test_file), 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(np.random.bytes(16000 * 2))
    
    # Set invalid transcription mode
    mock_app.transcription_mode = "invalid"
    
    # Should return empty string
    result = mock_app.transcribe_audio(str(test_file))
    assert result == "" 