"""Tests for wake word detection functionality."""

import pytest
import numpy as np
import openwakeword

from whisper_widget.app import SpeechToTextApp


class TestWakeWordDetection:
    """Test suite for wake word detection."""

    @pytest.fixture
    def speech_app(self):
        """Create a SpeechToTextApp instance for testing."""
        return SpeechToTextApp(wake_word="hey computer")

    def test_wake_word_initialization(self, speech_app):
        """Test that wake word detection is initialized correctly."""
        assert hasattr(speech_app, 'wake_word_detector')
        assert isinstance(speech_app.wake_word_detector, openwakeword.Model)
        assert speech_app.wake_word == "hey computer"
        assert not speech_app.wake_word_detected

    def test_wake_word_toggle(self, speech_app):
        """Test toggling wake word detection on and off."""
        # Initial state should be False
        assert not speech_app.wake_word_detected

        # Simulate toggling on
        speech_app.wake_word_detected = True
        assert speech_app.wake_word_detected

        # Simulate toggling off
        speech_app.wake_word_detected = False
        assert not speech_app.wake_word_detected

    def test_wake_word_prediction(self, speech_app, mocker):
        """Test wake word prediction mechanism."""
        # Create a mock audio chunk
        audio_chunk = np.random.randint(
            -32768, 32767, 
            size=16000, 
            dtype=np.int16
        )
        # Convert to float32 for prediction
        _ = audio_chunk.astype(np.float32) / 32768.0

        # Mock the wake word detector's predict method
        mock_predict = mocker.patch.object(
            speech_app.wake_word_detector, 
            'predict', 
            return_value=[(None, 0.6)]
        )

        # Simulate wake word detection loop
        speech_app.running = True
        speech_app.audio_buffer_ww.put(audio_chunk)

        # Call the wake word detection method
        speech_app._wake_word_loop()

        # Verify predictions were made
        mock_predict.assert_called_once()
        assert mock_predict.call_args[0][0].dtype == np.float32

    def test_wake_word_activation(self, speech_app, mocker):
        """Test that wake word activation triggers recording."""
        # Mock the start_recording method
        mock_start_recording = mocker.patch.object(
            speech_app, 
            'start_recording'
        )

        # Create a mock audio chunk 
        audio_chunk = np.random.randint(
            -32768, 32767, 
            size=16000, 
            dtype=np.int16
        )
        # Convert to float32 for prediction
        _ = audio_chunk.astype(np.float32) / 32768.0

        # Mock the wake word detector to return high confidence
        mocker.patch.object(
            speech_app.wake_word_detector, 
            'predict', 
            return_value=[(None, 0.8)]  # High confidence
        )

        # Ensure initial state
        assert not speech_app.wake_word_detected

        # Simulate wake word detection loop
        speech_app.running = True
        speech_app.audio_buffer_ww.put(audio_chunk)
        speech_app._wake_word_loop()

        # Verify recording was started and wake word was detected
        mock_start_recording.assert_called_once()
        assert speech_app.wake_word_detected

    def test_wake_word_low_confidence(self, speech_app, mocker):
        """Test that low confidence wake word does not trigger recording."""
        # Mock the start_recording method
        mock_start_recording = mocker.patch.object(
            speech_app, 
            'start_recording'
        )

        # Create a mock audio chunk
        audio_chunk = np.random.randint(
            -32768, 32767, 
            size=16000, 
            dtype=np.int16
        )
        # Convert to float32 for prediction
        _ = audio_chunk.astype(np.float32) / 32768.0

        # Mock the wake word detector to return low confidence
        mocker.patch.object(
            speech_app.wake_word_detector, 
            'predict', 
            return_value=[(None, 0.3)]  # Low confidence
        )

        # Ensure initial state
        assert not speech_app.wake_word_detected

        # Simulate wake word detection loop
        speech_app.running = True
        speech_app.audio_buffer_ww.put(audio_chunk)
        speech_app._wake_word_loop()

        # Verify recording was not started and wake word was not detected
        mock_start_recording.assert_not_called()
        assert not speech_app.wake_word_detected 