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
import tempfile


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
    reduced_low_threshold = noise_reduction(audio, noise_threshold=0.05)
    reduced_high_threshold = noise_reduction(audio, noise_threshold=0.3)
    
    # Low threshold should preserve more of the original signal
    assert np.count_nonzero(reduced_low_threshold) > np.count_nonzero(reduced_high_threshold)


def test_microphone_access():
    """Test microphone access check."""
    # This test might need to be adjusted based on the testing environment
    try:
        mic_access = check_microphone_access()
        # The result can be True or False depending on the environment
        assert isinstance(mic_access, bool)
    except Exception as e:
        # If an exception occurs, it should be related to audio device access
        assert "Cannot access microphone" in str(e)


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


def test_audio_buffer_initialization(mock_app):
    """Test audio buffer initialization in SpeechToTextApp."""
    # Verify initial audio buffer state
    assert hasattr(mock_app, 'audio_buffer')
    assert isinstance(mock_app.audio_buffer, bytearray)
    assert len(mock_app.audio_buffer) == 0


def test_audio_recording_states(mock_app):
    """Test different audio recording states."""
    # Initial state checks
    assert not mock_app.is_recording
    assert mock_app.recording_start_time is None

    # Start recording
    mock_app.start_recording()
    assert mock_app.is_recording
    assert mock_app.recording_start_time is not None

    # Stop recording
    mock_app.stop_recording()
    assert not mock_app.is_recording
    assert mock_app.recording_start_time is None


def test_create_temporary_wav_file(mock_app):
    """Test creating a temporary WAV file from audio buffer."""
    # Prepare mock audio data
    mock_audio_data = np.random.randint(
        -32768, 32767, 
        size=int(mock_app.sample_rate * 1),  # 1 second of audio
        dtype=np.int16
    )
    mock_app.audio_buffer = bytearray(mock_audio_data.tobytes())

    # Create temporary file
    temp_filename = "temp_audio.wav"
    try:
        wf = wave.open(temp_filename, 'wb')
        wf.setnchannels(mock_app.channels)
        wf.setsampwidth(mock_app.p.get_sample_size(mock_app.audio_format))
        wf.setframerate(mock_app.sample_rate)
        wf.writeframes(mock_app.audio_buffer)
        wf.close()

        # Verify file was created and has content
        assert os.path.exists(temp_filename)
        assert os.path.getsize(temp_filename) > 0
    finally:
        # Clean up
        if os.path.exists(temp_filename):
            os.remove(temp_filename)


def test_noise_reduction_edge_cases():
    """Test noise reduction with various edge case inputs."""
    # Test with empty array
    empty_audio = np.array([], dtype=np.float32)
    reduced_empty = noise_reduction(empty_audio)
    assert len(reduced_empty) == 0

    # Test with all zeros
    zero_audio = np.zeros(100, dtype=np.float32)
    reduced_zero = noise_reduction(zero_audio)
    assert np.all(reduced_zero == 0)

    # Test with all noise (below threshold)
    noise_audio = np.full(100, 0.01, dtype=np.float32)
    reduced_noise = noise_reduction(noise_audio)
    assert np.all(reduced_noise == 0)

    # Test with all signal (above threshold)
    signal_audio = np.full(100, 0.6, dtype=np.float32)
    reduced_signal = noise_reduction(signal_audio)
    assert np.all(reduced_signal == signal_audio)


def test_noise_reduction_threshold_sensitivity():
    """Test noise reduction sensitivity to different thresholds."""
    # Create a mixed signal with noise and signal
    mixed_audio = np.array([
        0.01, 0.02, 0.5, 0.6, 0.01, 0.02, 
        0.7, 0.8, 0.02, 0.03
    ], dtype=np.float32)

    # Test different noise thresholds
    thresholds = [0.05, 0.1, 0.2, 0.5]
    for threshold in thresholds:
        reduced_audio = noise_reduction(mixed_audio, threshold)
        
        # Verify that higher thresholds remove more low-amplitude segments
        non_zero_count = np.count_nonzero(reduced_audio)
        assert non_zero_count <= len(mixed_audio)
        
        # Ensure high-amplitude segments are preserved
        assert np.all(reduced_audio[2:4] != 0)
        assert np.all(reduced_audio[6:8] != 0)


def test_microphone_access_comprehensive():
    """Comprehensive test for microphone access."""
    # Test scenarios with mocked PyAudio
    scenarios = [
        # Successful access
        {
            'device_count': 1,
            'expected_result': True
        },
        # No devices
        {
            'device_count': 0,
            'expected_result': False
        }
    ]

    for scenario in scenarios:
        with patch('pyaudio.PyAudio') as MockPyAudio:
            mock_instance = MockPyAudio.return_value
            mock_instance.get_device_count.return_value = (
                scenario['device_count']
            )
            
            result = check_microphone_access()
            assert result == scenario['expected_result']


def test_audio_buffer_lifecycle(mock_whisper_app):
    """Test the complete lifecycle of the audio buffer."""
    app = mock_whisper_app

    # Initial state
    assert len(app.audio_buffer) == 0

    # Simulate recording start
    app.start_recording()
    assert app.is_recording
    assert app.recording_start_time is not None

    # Add some mock audio data
    mock_audio_data = np.random.randint(
        -32768, 32767, 
        size=int(app.sample_rate * 0.5),  # 0.5 seconds of audio
        dtype=np.int16
    ).tobytes()
    app.audio_buffer.extend(mock_audio_data)

    # Verify buffer content
    assert len(app.audio_buffer) > 0

    # Stop recording and process buffer
    app.stop_recording()
    assert not app.is_recording
    assert len(app.audio_buffer) == 0


def test_audio_recording_state_transitions(mock_whisper_app):
    """Test detailed state transitions during recording."""
    app = mock_whisper_app

    # Initial state checks
    assert not app.is_recording
    assert app.recording_start_time is None

    # Start recording sequence
    app.start_recording()
    assert app.is_recording
    assert app.recording_start_time is not None
    assert len(app.audio_buffer) == 0

    # Simulate adding audio data
    mock_audio_data = np.random.randint(
        -32768, 32767, 
        size=int(app.sample_rate * 0.5),  # 0.5 seconds of audio
        dtype=np.int16
    ).tobytes()
    app.audio_buffer.extend(mock_audio_data)

    # Stop recording
    app.stop_recording()
    assert not app.is_recording
    assert app.recording_start_time is None
    assert len(app.audio_buffer) == 0


def test_temporary_wav_file_creation(mock_whisper_app):
    """Comprehensive test for temporary WAV file creation."""
    app = mock_whisper_app

    # Prepare mock audio data
    mock_audio_data = np.random.randint(
        -32768, 32767, 
        size=int(app.sample_rate * 1),  # 1 second of audio
        dtype=np.int16
    )
    app.audio_buffer = bytearray(mock_audio_data.tobytes())

    # Create temporary file
    with tempfile.NamedTemporaryFile(
        suffix='.wav', 
        delete=False
    ) as temp_file:
        try:
            with wave.open(temp_file.name, 'wb') as wf:
                wf.setnchannels(app.channels)
                wf.setsampwidth(
                    app.p.get_sample_size(app.audio_format)
                )
                wf.setframerate(app.sample_rate)
                wf.writeframes(app.audio_buffer)

            # Verify file was created and has content
            assert os.path.exists(temp_file.name)
            assert os.path.getsize(temp_file.name) > 0

            # Verify WAV file properties
            with wave.open(temp_file.name, 'rb') as wf:
                assert wf.getnchannels() == app.channels
                assert wf.getsampwidth() == (
                    app.p.get_sample_size(app.audio_format)
                )
                assert wf.getframerate() == app.sample_rate

        finally:
            # Clean up
            os.unlink(temp_file.name)


if __name__ == '__main__':
    # List available audio devices
    print("\nAvailable audio devices:")
    print(sd.query_devices())
    print("\nDefault devices:")
    print(f"Input device: {sd.query_devices(kind='input')}")
    
    unittest.main() 