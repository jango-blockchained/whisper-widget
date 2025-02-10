"""Audio utility functions for the whisper widget."""

import logging
import numpy as np
import pyaudio


def noise_reduction(
    audio_data,
    sample_rate=None,
    threshold=0.2,
    frame_length=2048,
    hop_length=512
):
    """
    Reduce noise in audio data with improved handling and adaptive thresholding.
    
    Args:
        audio_data (np.ndarray or bytes): Input audio data
        sample_rate (int, optional): Sampling rate of the audio
        threshold (float, optional): Base noise reduction threshold
        frame_length (int, optional): Length of each frame for processing
        hop_length (int, optional): Number of samples between frames
    
    Returns:
        np.ndarray or bytes: Noise-reduced audio data
    
    Raises:
        ValueError: If audio_data is invalid or empty
    """
    if audio_data is None:
        raise ValueError("Audio data cannot be None")

    # Convert bytes to numpy array if needed
    was_bytes = isinstance(audio_data, bytes)
    if was_bytes:
        try:
            audio_data = np.frombuffer(audio_data, dtype=np.int16)
            audio_data = audio_data.astype(np.float32)
        except ValueError as e:
            raise ValueError(f"Failed to convert audio bytes to array: {e}")

    # Handle empty array
    if len(audio_data) == 0:
        raise ValueError("Audio data is empty")

    # Prevent division by zero and handle zero arrays
    if np.all(audio_data == 0):
        return (
            np.zeros_like(audio_data, dtype=np.float32) 
            if not was_bytes 
            else bytes(len(audio_data))
        )

    try:
        # Normalize audio data
        max_abs = np.max(np.abs(audio_data))
        if max_abs > 0:
            audio_normalized = audio_data / max_abs
        else:
            audio_normalized = audio_data

        # Calculate adaptive threshold using local statistics
        frames = np.array([
            audio_normalized[i:i + frame_length]
            for i in range(0, len(audio_normalized), hop_length)
            if i + frame_length <= len(audio_normalized)
        ])
        
        if len(frames) > 0:
            # Calculate local energy for each frame
            frame_energy = np.mean(frames ** 2, axis=1)
            # Adaptive threshold based on local energy statistics
            adaptive_threshold = threshold * (
                np.mean(frame_energy) + np.std(frame_energy)
            )
            
            # Apply adaptive thresholding
            noise_mask = np.abs(audio_normalized) < adaptive_threshold
            audio_reduced = audio_normalized.copy()
            audio_reduced[noise_mask] = 0.0
            
            # Apply smooth transitions
            transition_length = min(256, len(audio_reduced) // 100)
            if transition_length > 0:
                window = np.hanning(transition_length * 2)
                for i in range(1, len(noise_mask) - 1):
                    if noise_mask[i] != noise_mask[i - 1]:
                        start = max(0, i - transition_length)
                        end = min(len(audio_reduced), i + transition_length)
                        audio_reduced[start:end] *= window[:end-start]
        else:
            # Fallback to simple thresholding for short audio
            noise_mask = np.abs(audio_normalized) < threshold
            audio_reduced = audio_normalized.copy()
            audio_reduced[noise_mask] = 0.0

        # Convert back to original type if input was bytes
        if was_bytes:
            audio_reduced = (
                audio_reduced * np.iinfo(np.int16).max
            ).astype(np.int16).tobytes()

        return audio_reduced

    except Exception as e:
        logging.error(f"Error in noise reduction: {e}")
        # Return original audio data if processing fails
        return audio_data if not was_bytes else audio_data.tobytes()


def check_microphone_access() -> bool:
    """
    Check microphone access with improved device detection.
    
    Returns:
        bool: True if input devices are available, False otherwise
    """
    try:
        p = pyaudio.PyAudio()
        device_count = p.get_device_count()
        input_devices = [
            p.get_device_info_by_index(i)
            for i in range(device_count)
            if p.get_device_info_by_index(i)['maxInputChannels'] > 0
        ]
        p.terminate()
        return len(input_devices) > 0
    except Exception:
        return False 