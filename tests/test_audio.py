"""Tests for audio processing and recording functionality."""

import pytest
from unittest.mock import MagicMock, patch
import numpy as np
import wave
import os
import tempfile
import sounddevice as sd
import pyaudio

from whisper_widget.audio.processor import AudioProcessor
from whisper_widget.utils.audio_utils import noise_reduction, check_microphone_access


def test_check_microphone_access(mock_audio):
    """Test microphone access check with mock audio."""
    assert check_microphone_access() is True


def test_check_microphone_access_no_device():
    """Test microphone access check with no device available."""
    with patch('pyaudio.PyAudio') as mock_pyaudio:
        mock_instance = mock_pyaudio.return_value
        mock_instance.get_device_count.return_value = 0
        assert check_microphone_access() is False


def test_noise_reduction():
    """Test noise reduction function."""
    # Create a sample audio array with noise
    audio = np.array([0.01, 0.02, 0.5, 0.6, 0.01, 0.02], dtype=np.float32)
    
    # Apply noise reduction
    reduced_audio = noise_reduction(audio)
    
    # Check that low-amplitude segments are zeroed out
    assert np.all(reduced_audio[0:2] == 0)
    assert np.all(reduced_audio[2:4] != 0)
    assert np.all(reduced_audio[4:] == 0)


def test_noise_reduction_different_thresholds():
    """Test noise reduction with different thresholds."""
    # Create a sample audio array
    audio = np.array([0.1, 0.2, 0.5, 0.6, 0.1, 0.2], dtype=np.float32)
    
    # Test with different noise thresholds
    reduced_low = noise_reduction(audio, threshold=0.05)
    reduced_high = noise_reduction(audio, threshold=0.3)
    
    # Low threshold should preserve more of the original signal
    assert np.count_nonzero(reduced_low) > np.count_nonzero(reduced_high)


def test_audio_processor_initialization(mock_audio):
    """Test AudioProcessor initialization."""
    processor = AudioProcessor()
    
    # Check default settings
    assert processor.sample_rate == 16000
    assert processor.channels == 1
    assert processor.chunk_duration == 0.05
    assert processor.vad_sensitivity == 2
    assert processor.on_speech_detected is None
    
    # Check audio components
    assert hasattr(processor, 'p')
    assert hasattr(processor, 'stream')
    assert hasattr(processor, 'vad')


def test_audio_processor_recording(mock_audio):
    """Test AudioProcessor recording functionality."""
    processor = AudioProcessor()
    
    # Test start recording
    processor.start_recording()
    assert processor.is_recording
    assert processor.audio_thread is not None
    assert processor.audio_thread.is_alive()
    
    # Test stop recording
    processor.stop_recording()
    assert not processor.is_recording
    assert not processor.audio_thread.is_alive()


def test_audio_processor_speech_detection(mock_audio):
    """Test speech detection in AudioProcessor."""
    # Create processor with mock speech detection callback
    mock_callback = MagicMock()
    processor = AudioProcessor(on_speech_detected=mock_callback)
    
    # Create test audio data
    test_data = np.random.randint(-32768, 32767, 16000, dtype=np.int16)
    test_bytes = test_data.tobytes()
    
    # Process audio with speech
    with patch('webrtcvad.Vad.is_speech', return_value=True):
        processor._process_audio_chunk(test_bytes)
        processor._update_speech_state(True, test_bytes)
        processor._update_speech_state(False, test_bytes)
    
    # Verify callback was called
    assert mock_callback.called


def test_audio_processor_save_audio(mock_audio):
    """Test saving audio to WAV file."""
    processor = AudioProcessor()
    
    # Create test audio data
    test_data = np.random.randint(-32768, 32767, 16000, dtype=np.int16)
    test_bytes = test_data.tobytes()
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        try:
            # Save audio
            success = processor.save_audio(test_bytes, temp_file.name)
            assert success
            
            # Verify file exists and has content
            assert os.path.exists(temp_file.name)
            assert os.path.getsize(temp_file.name) > 0
            
            # Verify WAV file properties
            with wave.open(temp_file.name, 'rb') as wf:
                assert wf.getnchannels() == processor.channels
                assert wf.getframerate() == processor.sample_rate
                assert wf.getsampwidth() == 2  # 16-bit audio
                
        finally:
            os.unlink(temp_file.name)


def test_audio_processor_error_handling(mock_audio):
    """Test error handling in AudioProcessor."""
    processor = AudioProcessor()
    
    # Test invalid audio chunk
    processor._process_audio_chunk(b'invalid data')
    assert not processor.is_speech
    
    # Test VAD error
    with patch('webrtcvad.Vad.is_speech', side_effect=Exception("VAD error")):
        processor._process_audio_chunk(np.random.bytes(1600))
        assert not processor.is_speech


def test_audio_quality_checks(mock_audio):
    """Test audio quality checking in AudioProcessor."""
    processor = AudioProcessor()
    
    # Test silence detection
    silent_audio = np.zeros(1600, dtype=np.int16)
    assert not processor._check_audio_quality(silent_audio)
    
    # Test clipping detection
    clipped_audio = np.full(1600, 32767, dtype=np.int16)
    assert not processor._check_audio_quality(clipped_audio)
    
    # Test DC offset detection
    dc_audio = np.full(1600, 16000, dtype=np.int16)
    assert not processor._check_audio_quality(dc_audio)
    
    # Test good quality audio
    good_audio = np.random.randint(-16000, 16000, 1600, dtype=np.int16)
    assert processor._check_audio_quality(good_audio)


if __name__ == '__main__':
    # List available audio devices
    print("\nAvailable audio devices:")
    print(sd.query_devices())
    print("\nDefault devices:")
    print(f"Input device: {sd.query_devices(kind='input')}")
    
    unittest.main() 