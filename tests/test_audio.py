import pytest
from unittest.mock import MagicMock, patch
import numpy as np
import wave
import os
from whisper_widget.app import (
    check_microphone_access,
    noise_reduction,
    SpeechToTextApp
)
import unittest
import sounddevice as sd
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
        return np.random.bytes(chunk_size * 2)  # 16-bit audio = 2 bytes per sample


class MockPyAudio:
    def __init__(self):
        self.streams = []
        self.sample_rate = 16000
        self.audio_format = pyaudio.paInt16

    def get_sample_size(self, format_type):
        return 2  # 16-bit audio = 2 bytes per sample

    def open(
        self,
        format=None,
        channels=None,
        rate=None,
        input=None,
        frames_per_buffer=None,
        stream_callback=None,
        start=True,
        input_device_index=None,
        **kwargs
    ):
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
        mock_instance.get_default_input_device_info.side_effect = (
            OSError("No device found")
        )
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
    with patch('whisper_widget.app.WhisperModel') as mock_model, \
         patch('whisper_widget.app.webrtcvad.Vad') as mock_vad:
        
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
        app.sample_rate = 16000  # Add sample rate
        app.audio_format = pyaudio.paInt16  # Add audio format
        app.channels = 1  # Add channels
        app.p = MockPyAudio()  # Add PyAudio instance
        
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


def test_noise_reduction_with_threshold():
    """Test noise reduction with different thresholds."""
    # Create test audio data
    test_data = np.random.randint(-32768, 32767, 1000, dtype=np.int16).tobytes()
    sample_rate = 16000

    # Test with different thresholds
    thresholds = [0.0, 0.1, 0.5, 1.0]
    for threshold in thresholds:
        result = noise_reduction(test_data, sample_rate, threshold)
        assert isinstance(result, bytes)
        assert len(result) == len(test_data)


def test_speech_detection_parameters(mock_whisper_app):
    """Test speech detection parameter initialization and updates."""
    app = mock_whisper_app

    # Check default values
    assert app.settings['min_speech_duration'] == 0.5
    assert app.settings['max_silence_duration'] == 1.0
    assert app.settings['min_audio_length'] == 1.0
    assert app.settings['speech_threshold'] == 0.5
    assert app.settings['silence_threshold'] == 10
    assert app.settings['speech_start_chunks'] == 2
    assert app.settings['noise_reduce_threshold'] == 0.1

    # Test updating parameters
    app.update_setting('min_speech_duration', 1.0)
    assert app.settings['min_speech_duration'] == 1.0

    app.update_setting('max_silence_duration', 2.0)
    assert app.settings['max_silence_duration'] == 2.0

    app.update_setting('speech_start_chunks', 3)
    assert app.settings['speech_start_chunks'] == 3


def test_auto_detection_start_stop(mock_whisper_app):
    """Test auto-detection start and stop conditions."""
    app = mock_whisper_app
    app.settings['auto_detect_speech'] = True
    app.settings['speech_start_chunks'] = 2
    app.settings['min_speech_duration'] = 0.5
    app.settings['max_silence_duration'] = 1.0
    app.recording = False  # Ensure we start not recording

    # Create a temporary WAV file for testing
    test_file = "test_audio.wav"
    sample_rate = 16000
    duration = 2  # seconds
    samples = np.random.randint(
        -32768,
        32767,
        duration * sample_rate,
        dtype=np.int16
    )
    
    with wave.open(test_file, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())

    try:
        # Create a repeating pattern for speech detection
        base_pattern = [True, True, False, False]
        pattern_repeats = 10  # Repeat pattern 10 times
        speech_pattern = base_pattern * pattern_repeats
        
        def speech_detector(data, rate):
            try:
                return speech_pattern.pop(0)
            except IndexError:
                return False

        app.vad.is_speech = speech_detector

        # Process audio in chunks
        chunk_size = int(sample_rate * 0.03)  # 30ms chunks
        speech_chunks = 0
        with wave.open(test_file, 'rb') as wf:
            while True:
                data = wf.readframes(chunk_size)
                if not data:
                    break
                
                # Simulate audio processing
                is_speech = app.vad.is_speech(data, sample_rate)
                if is_speech:
                    speech_chunks += 1
                else:
                    speech_chunks = max(0, speech_chunks - 1)
                
                # Check recording state
                should_record = (
                    app.settings['auto_detect_speech'] and 
                    speech_chunks >= app.settings['speech_start_chunks']
                )
                
                if should_record:
                    if not app.recording:
                        app.start_recording()
                    app.audio_buffer.extend(data)
                elif app.recording:
                    app.stop_recording()

        # Verify final state
        assert not app.recording, "Should stop recording at end"
        assert len(app.audio_buffer) == 0, "Should have processed buffer"

    finally:
        # Cleanup
        if os.path.exists(test_file):
            os.remove(test_file)


def test_speech_silence_duration_tracking(mock_whisper_app):
    """Test tracking of speech and silence durations."""
    app = mock_whisper_app
    app.settings['min_speech_duration'] = 0.5
    app.settings['max_silence_duration'] = 1.0
    app.recording = False  # Ensure we start not recording
    
    # Create test audio data
    sample_rate = 16000
    chunk_duration = 0.03  # 30ms chunks
    chunk_size = int(sample_rate * chunk_duration)
    
    # Simulate alternating speech/silence pattern
    # 600ms speech + 1200ms silence
    speech_pattern = [True] * 20 + [False] * 40
    app.vad.is_speech = MagicMock(side_effect=speech_pattern)
    
    # Process chunks
    silence_duration = 0.0
    for i, is_speech in enumerate(speech_pattern):
        data = np.random.bytes(chunk_size * 2)  # 16-bit audio
        
        if is_speech:
            if not app.recording:
                app.start_recording()
            silence_duration = 0.0
        else:
            silence_duration += chunk_duration
            
        if app.recording:
            app.audio_buffer.extend(data)
            
            # Check if we should stop
            if silence_duration >= app.settings['max_silence_duration']:
                app.stop_recording()
                assert not app.recording, "Should stop after max silence duration"
    
    # Cleanup
    if app.recording:
        app.stop_recording()


def test_noise_reduction_integration(mock_whisper_app):
    """Test noise reduction integration in audio processing."""
    app = mock_whisper_app
    app.settings['noise_reduce_threshold'] = 0.2

    # Create test audio data
    sample_rate = 16000
    chunk_size = int(sample_rate * 0.03)
    test_data = np.random.bytes(chunk_size * 2)

    # Mock noise_reduction function
    with patch('whisper_widget.app.noise_reduction') as mock_noise_reduce:
        mock_noise_reduce.return_value = test_data

        # Process audio chunk with noise reduction
        app.process_audio_chunk(test_data)

        # Verify noise reduction was called with correct parameters
        mock_noise_reduce.assert_called_with(
            test_data,
            app.sample_rate,
            threshold=app.settings['noise_reduce_threshold']
        ) 


class TestAudioInput(unittest.TestCase):
    def test_audio_input(self):
        """Test if we can capture audio from the default input device"""
        duration = 1  # seconds
        sample_rate = 16000
        
        # Record some audio
        recording = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype=np.float32
        )
        sd.wait()  # Wait until recording is finished
        
        # Check if we got any signal (non-zero values)
        self.assertTrue(np.any(recording != 0), "No audio signal detected")
        
        # Print max amplitude to see if we're getting meaningful input
        print(f"Maximum audio amplitude: {np.max(np.abs(recording))}")


if __name__ == '__main__':
    # List available audio devices
    print("\nAvailable audio devices:")
    print(sd.query_devices())
    print("\nDefault devices:")
    print(f"Input device: {sd.query_devices(kind='input')}")
    
    unittest.main() 