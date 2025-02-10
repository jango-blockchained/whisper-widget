"""Audio processing module for recording and processing audio input."""

import logging
import threading
import wave
from typing import Optional, Callable
import numpy as np
import pyaudio
import webrtcvad

from ..utils.audio_utils import noise_reduction


class AudioProcessor:
    """Handles audio recording and processing."""

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_duration: float = 0.05,
        vad_sensitivity: int = 2,
        on_speech_detected: Optional[Callable[[bytes], None]] = None
    ):
        """Initialize audio processor with specified settings."""
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_duration = chunk_duration
        self.chunk_size = int(sample_rate * chunk_duration)
        self.vad_sensitivity = vad_sensitivity
        self.on_speech_detected = on_speech_detected

        # Audio processing settings
        self.settings = {
            'min_speech_duration': 0.5,
            'max_silence_duration': 1.0,
            'min_audio_length': 0.5,
            'speech_threshold': 0.5,
            'silence_threshold': 15,
            'speech_start_chunks': 4,
            'noise_reduce_threshold': 0.15
        }

        # Initialize state
        self.is_recording = False
        self.audio_thread = None
        self.audio_buffer = bytearray()
        self.processed_chunks = []
        self.is_speech = False
        self.speech_start = None
        self.silence_start = None

        # Initialize audio components
        self.init_audio()

    def init_audio(self) -> None:
        """Initialize audio components with error handling."""
        try:
            # Initialize PyAudio
            try:
                self.p = pyaudio.PyAudio()
            except OSError as e:
                logging.error(f"Failed to initialize PyAudio: {e}")
                raise RuntimeError("Could not initialize audio system")

            # Get device info
            try:
                device_count = self.p.get_device_count()
                if device_count == 0:
                    raise RuntimeError("No audio devices found")
                    
                # Find best available input device
                input_devices = [
                    self.p.get_device_info_by_index(i)
                    for i in range(device_count)
                    if self.p.get_device_info_by_index(i)['maxInputChannels'] > 0
                ]
                
                if not input_devices:
                    raise RuntimeError("No input devices available")
                    
                # Select device with highest number of input channels
                selected_device = max(
                    input_devices,
                    key=lambda x: x['maxInputChannels']
                )
                self.input_device_index = selected_device['index']
                
            except OSError as e:
                logging.error(f"Error accessing audio devices: {e}")
                raise RuntimeError("Could not access audio devices")

            # Initialize VAD
            try:
                self.vad = webrtcvad.Vad(self.vad_sensitivity)
            except Exception as e:
                logging.error(f"Failed to initialize VAD: {e}")
                self.vad = None

            # Initialize audio stream
            max_retries = 3
            retry_count = 0
            while retry_count < max_retries:
                try:
                    self.stream = self.p.open(
                        format=pyaudio.paInt16,
                        channels=self.channels,
                        rate=self.sample_rate,
                        input=True,
                        frames_per_buffer=self.chunk_size,
                        input_device_index=self.input_device_index
                    )
                    # Test the stream
                    test_data = self.stream.read(self.chunk_size)
                    if not test_data:
                        raise RuntimeError("Stream read test failed")
                    break
                    
                except (OSError, IOError) as e:
                    retry_count += 1
                    msg = f"Attempt {retry_count} to open audio stream failed: {e}"
                    logging.warning(msg)
                    if retry_count == max_retries:
                        msg = "Failed to initialize audio stream after retries"
                        raise RuntimeError(msg)
                    threading.Event().wait(1.0)  # Wait before retrying

            logging.info("Audio system initialized successfully")
            
        except Exception as e:
            logging.error(f"Audio initialization failed: {e}")
            self.cleanup()
            raise

    def start_recording(self) -> None:
        """Start recording audio."""
        if not hasattr(self, 'stream') or not self.stream:
            logging.error("Audio stream not initialized")
            return

        self.is_recording = True
        self.audio_buffer.clear()
        self.processed_chunks = []
        self.is_speech = False
        self.speech_start = None
        self.silence_start = None

        # Start audio processing thread
        self.audio_thread = threading.Thread(
            target=self._audio_processing_loop,
            daemon=True
        )
        self.audio_thread.start()

    def stop_recording(self) -> None:
        """Stop recording and reset state."""
        self.is_recording = False
        if self.audio_thread:
            self.audio_thread.join()
        self.audio_buffer.clear()

    def cleanup(self) -> None:
        """Clean up audio resources."""
        try:
            if hasattr(self, 'stream') and self.stream:
                try:
                    self.stream.stop_stream()
                    self.stream.close()
                except Exception as e:
                    logging.error(f"Error closing audio stream: {e}")
                    
            if hasattr(self, 'p') and self.p:
                try:
                    self.p.terminate()
                except Exception as e:
                    logging.error(f"Error terminating PyAudio: {e}")
                    
        except Exception as e:
            logging.error(f"Error in audio cleanup: {e}")
        finally:
            self.stream = None
            self.p = None
            self.vad = None

    def _audio_processing_loop(self) -> None:
        """Main audio processing loop."""
        error_count = 0
        max_errors = 5
        last_error_time = 0
        error_reset_interval = 60

        while self.is_recording:
            try:
                current_time = threading.Event().time()
                if current_time - last_error_time > error_reset_interval:
                    error_count = 0

                # Read audio chunk
                try:
                    audio_chunk = self.stream.read(
                        self.chunk_size,
                        exception_on_overflow=False
                    )
                except (OSError, IOError) as e:
                    error_count += 1
                    last_error_time = current_time
                    logging.warning(f"Error reading audio chunk: {e}")
                    
                    if error_count >= max_errors:
                        msg = "Too many audio read errors, stopping recording"
                        logging.error(msg)
                        self.stop_recording()
                        break
                        
                    threading.Event().wait(0.1)
                    continue

                # Validate audio chunk
                if not audio_chunk or len(audio_chunk) != self.chunk_size * 2:
                    logging.warning("Invalid audio chunk received")
                    continue

                # Process audio chunk
                self._process_audio_chunk(audio_chunk)

            except Exception as e:
                error_count += 1
                last_error_time = threading.Event().time()
                logging.error(f"Error in audio processing loop: {e}")
                
                if error_count >= max_errors:
                    msg = "Too many errors in audio loop, stopping recording"
                    logging.error(msg)
                    self.stop_recording()
                    break
                    
                threading.Event().wait(0.1)

    def _process_audio_chunk(self, audio_chunk: bytes) -> None:
        """Process a single audio chunk."""
        try:
            # Convert to numpy array for analysis
            audio_data = np.frombuffer(audio_chunk, dtype=np.int16)

            # Check audio quality
            if self._check_audio_quality(audio_data):
                # Apply noise reduction
                try:
                    processed_chunk = noise_reduction(
                        audio_chunk,
                        self.sample_rate,
                        self.settings['noise_reduce_threshold']
                    )
                except Exception as e:
                    logging.warning(f"Error in noise reduction: {e}")
                    processed_chunk = audio_chunk

                # Detect speech
                is_speech = False
                if self.vad:
                    try:
                        is_speech = self.vad.is_speech(
                            processed_chunk,
                            self.sample_rate
                        )
                    except Exception as e:
                        logging.warning(f"VAD error: {e}")
                        is_speech = self._detect_speech_energy(audio_data)
                else:
                    is_speech = self._detect_speech_energy(audio_data)

                # Update speech state
                self._update_speech_state(is_speech, processed_chunk)

        except Exception as e:
            logging.error(f"Error processing audio chunk: {e}")

    def _check_audio_quality(self, audio_data: np.ndarray) -> bool:
        """Check audio quality metrics."""
        try:
            # Check for silence/low volume
            rms = np.sqrt(np.mean(audio_data**2))
            if rms < 100:  # Arbitrary threshold
                return False

            # Check for clipping
            max_amplitude = np.max(np.abs(audio_data))
            if max_amplitude > 0.95 * np.iinfo(np.int16).max:
                return False

            # Check for DC offset
            mean_amplitude = np.mean(audio_data)
            if abs(mean_amplitude) > 1000:  # Arbitrary threshold
                return False

            return True
            
        except Exception as e:
            logging.warning(f"Error in audio quality check: {e}")
            return True

    def _detect_speech_energy(self, audio_data: np.ndarray) -> bool:
        """Detect speech using energy-based method."""
        try:
            energy = np.mean(audio_data**2)
            return energy > self.settings['speech_threshold']
        except Exception as e:
            logging.warning(f"Error in energy-based speech detection: {e}")
            return False

    def _update_speech_state(
        self,
        is_speech: bool,
        processed_chunk: bytes
    ) -> None:
        """Update speech detection state and handle transitions."""
        try:
            current_time = threading.Event().time()
            
            if is_speech:
                if not self.is_speech:  # Speech start
                    self.is_speech = True
                    self.speech_start = current_time
                    self.silence_start = None
                    self.processed_chunks = []
                
                self.processed_chunks.append(processed_chunk)
                
            else:  # No speech
                if self.is_speech:  # Potential speech end
                    if not self.silence_start:
                        self.silence_start = current_time
                        
                    silence_duration = current_time - self.silence_start
                    max_silence = self.settings['max_silence_duration']
                    if silence_duration >= max_silence:
                        # Speech segment complete
                        min_speech = self.settings['min_speech_duration']
                        if current_time - self.speech_start >= min_speech:
                            self._handle_speech_segment()
                        
                        self.is_speech = False
                        self.speech_start = None
                        self.processed_chunks = []
                    else:
                        # Still in speech segment, keep chunk
                        self.processed_chunks.append(processed_chunk)
                
                else:  # Continued silence
                    self.silence_start = current_time
                    
        except Exception as e:
            logging.error(f"Error updating speech state: {e}")

    def _handle_speech_segment(self) -> None:
        """Process a complete speech segment."""
        if not self.processed_chunks:
            return

        try:
            # Combine all chunks
            audio_data = b''.join(self.processed_chunks)
            
            # Save to WAV file if needed
            if self.on_speech_detected:
                self.on_speech_detected(audio_data)
                
        except Exception as e:
            logging.error(f"Error handling speech segment: {e}")

    def save_audio(self, audio_data: bytes, filename: str) -> bool:
        """Save audio data to a WAV file."""
        try:
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.p.get_sample_size(pyaudio.paInt16))
                wf.setframerate(self.sample_rate)
                wf.writeframes(audio_data)
            return True
        except Exception as e:
            logging.error(f"Error saving audio file: {e}")
            return False

    def update_settings(self, **kwargs) -> None:
        """Update audio processing settings."""
        for key, value in kwargs.items():
            if key in self.settings:
                self.settings[key] = value 