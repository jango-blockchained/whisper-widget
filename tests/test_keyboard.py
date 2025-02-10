import pytest
from unittest.mock import MagicMock, patch
from pynput.keyboard import Key
from app import SpeechToTextApp


@pytest.fixture
def mock_app(mock_audio):
    """Create a mock app for keyboard testing."""
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


def test_f9_key_toggle(mock_app):
    """Test F9 key toggling recording."""
    # Initially not recording
    assert mock_app.recording is False
    
    # Press F9 to start recording
    mock_app.on_key_press(Key.f9)
    assert mock_app.recording is True
    mock_app.indicator.set_icon.assert_called()
    
    # Press F9 again to stop recording
    mock_app.on_key_press(Key.f9)
    assert mock_app.recording is False
    mock_app.indicator.set_icon.assert_called()


def test_other_keys_no_effect(mock_app):
    """Test that other keys don't affect recording."""
    initial_state = mock_app.recording
    
    # Press various other keys
    for key in [Key.enter, Key.space, Key.esc]:
        mock_app.on_key_press(key)
        assert mock_app.recording == initial_state


def test_key_press_error_handling(mock_app):
    """Test error handling in key press handler."""
    # Simulate an error by passing an invalid key
    mock_app.on_key_press(None)
    assert mock_app.recording is False  # Should not change state
    
    # Test with an exception-raising key
    class ErrorKey:
        def __eq__(self, other):
            raise Exception("Test error")
    
    mock_app.on_key_press(ErrorKey())
    assert mock_app.recording is False  # Should not change state 