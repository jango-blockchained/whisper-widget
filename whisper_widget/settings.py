"""Settings management module."""

import json
from pathlib import Path
from typing import Any, Dict


def load_settings() -> Dict[str, Any]:
    """Load settings from config file."""
    config_dir = Path.home() / '.config' / 'whisper-widget'
    config_file = config_dir / 'config.json'
    default_settings: Dict[str, Any] = {
        'transcription_mode': 'continuous',  # continuous or clipboard
        'model_size': 'base',  # tiny, base, small, medium, large
        'language': 'en',  # language code
        'vad_sensitivity': 3,  # 1-3
        'auto_detect_speech': True,
        'add_punctuation': True,
        'min_speech_duration': 0.5,  # seconds
        'max_silence_duration': 1.0,  # seconds
        'min_audio_length': 1.0,  # seconds
        'speech_threshold': 0.5,  # 0.0-1.0
        'silence_threshold': 10,  # silence threshold
        'speech_start_chunks': 2,  # consecutive speech chunks to start
        'noise_reduce_threshold': 0.1,  # 0.0-1.0
        'sample_rate': 16000,  # Fixed for Whisper
    }
    
    if config_file.exists():
        try:
            with open(config_file) as f:
                return {**default_settings, **json.load(f)}
        except Exception as e:
            print(f"Error loading settings: {e}")
            return default_settings
    return default_settings


def save_settings(settings: Dict[str, Any]) -> None:
    """Save settings to config file."""
    config_dir = Path.home() / '.config' / 'whisper-widget'
    config_file = config_dir / 'config.json'
    config_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(config_file, 'w') as f:
            json.dump(settings, f, indent=4)
    except Exception as e:
        print(f"Error saving settings: {e}") 