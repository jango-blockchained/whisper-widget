"""Utility functions for audio processing."""

import numpy as np


def noise_reduction(
    audio_data: bytes,
    sample_rate: int,
    threshold: float = 0.1
) -> bytes:
    """
    Apply noise reduction to the audio data.
    
    Args:
        audio_data: Raw audio data bytes
        sample_rate: Audio sample rate in Hz
        threshold: Noise reduction sensitivity (0.0-1.0)
        
    Returns:
        Processed audio data bytes
    """
    # Convert bytes to numpy array
    audio_array = np.frombuffer(audio_data, dtype=np.int16)
    
    # Calculate maximum amplitude
    max_amplitude = np.max(np.abs(audio_array))
    
    # Apply noise gate
    noise_gate = max_amplitude * threshold
    audio_array[np.abs(audio_array) < noise_gate] = 0
    
    # Convert back to bytes
    return audio_array.tobytes() 