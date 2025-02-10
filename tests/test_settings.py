import json
import pytest
from pathlib import Path
from app import load_settings, save_settings


def test_load_settings_default(temp_config_dir):
    """Test loading settings when no config file exists."""
    settings = load_settings()
    assert settings['transcription_mode'] == 'continuous'
    assert settings['model_size'] == 'base'
    assert settings['language'] == 'en'
    assert settings['vad_sensitivity'] == 3
    assert settings['auto_detect_speech'] is True
    assert settings['add_punctuation'] is True
    assert settings['sample_rate'] == 16000
    assert settings['silence_threshold'] == 10
    assert settings['min_speech_duration'] == 0.5
    assert settings['max_silence_duration'] == 1.0
    assert settings['min_audio_length'] == 1.0
    assert settings['speech_threshold'] == 0.5
    assert settings['speech_start_chunks'] == 2
    assert settings['noise_reduce_threshold'] == 0.1


def test_save_and_load_settings(temp_config_dir, sample_settings):
    """Test saving and loading settings."""
    # Modify some settings
    modified_settings = sample_settings.copy()
    modified_settings.update({
        'transcription_mode': 'clipboard',
        'model_size': 'small',
        'language': 'es',
        'vad_sensitivity': 2,
        'min_speech_duration': 1.0,
        'max_silence_duration': 2.0,
        'speech_start_chunks': 3,
        'noise_reduce_threshold': 0.2
    })
    
    # Save the settings
    save_settings(modified_settings)
    
    # Load the settings back
    loaded_settings = load_settings()
    
    # Verify the loaded settings match what we saved
    assert loaded_settings == modified_settings


def test_save_settings_creates_config_dir(temp_config_dir):
    """Test that save_settings creates the config directory if it doesn't exist."""
    config_dir = Path.home() / '.config' / 'whisper-widget'
    assert not config_dir.exists()
    
    save_settings({'test': 'value'})
    
    assert config_dir.exists()
    assert config_dir.is_dir()


def test_load_settings_invalid_json(temp_config_dir):
    """Test loading settings with invalid JSON in config file."""
    config_dir = Path.home() / '.config' / 'whisper-widget'
    config_dir.mkdir(parents=True)
    config_file = config_dir / 'config.json'
    
    # Write invalid JSON
    config_file.write_text('invalid json')
    
    # Should return default settings
    settings = load_settings()
    assert settings['transcription_mode'] == 'continuous'
    assert settings['model_size'] == 'base'


def test_save_settings_permission_error(temp_config_dir, monkeypatch):
    """Test saving settings when there's a permission error."""
    def mock_open(*args, **kwargs):
        raise PermissionError("Permission denied")
    
    monkeypatch.setattr('builtins.open', mock_open)
    
    # Should not raise an exception
    save_settings({'test': 'value'})


def test_load_settings_merges_defaults(temp_config_dir):
    """Test that load_settings merges defaults with existing settings."""
    # Save partial settings
    partial_settings = {
        'transcription_mode': 'clipboard',
        'model_size': 'small'
    }
    save_settings(partial_settings)
    
    # Load settings - should include defaults for missing values
    settings = load_settings()
    assert settings['transcription_mode'] == 'clipboard'  # From saved
    assert settings['model_size'] == 'small'  # From saved
    assert settings['language'] == 'en'  # Default
    assert settings['vad_sensitivity'] == 3  # Default 