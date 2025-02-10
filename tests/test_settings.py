"""Tests for settings management module."""

import json
import logging
import os
import pytest
from whisper_widget.settings import (
    SettingsLoadError,
    SettingsSaveError,
    setup_logging,
    validate_settings,
    load_settings,
    save_settings,
    get_setting
)


@pytest.fixture
def temp_config_dir(tmp_path):
    """Create a temporary config directory."""
    old_home = os.environ.get('HOME')
    os.environ['HOME'] = str(tmp_path)
    yield tmp_path
    if old_home:
        os.environ['HOME'] = old_home


@pytest.fixture
def valid_settings():
    """Return a dictionary of valid settings."""
    return {
        'transcription_mode': 'continuous',
        'model_size': 'base',
        'language': 'en',
        'vad_sensitivity': 3,
        'auto_detect_speech': True,
        'add_punctuation': True,
        'min_speech_duration': 0.5,
        'max_silence_duration': 1.0,
        'min_audio_length': 1.0,
        'speech_threshold': 0.5,
        'silence_threshold': 10,
        'speech_start_chunks': 2,
        'noise_reduce_threshold': 0.1,
        'sample_rate': 16000,
    }


def test_setup_logging(temp_config_dir):
    """Test logging setup."""
    setup_logging()
    log_file = temp_config_dir / '.config' / 'whisper-widget' / 'app.log'
    assert log_file.exists()
    
    logger = logging.getLogger('whisper_widget.settings')
    logger.info('Test log message')
    
    with open(log_file) as f:
        log_content = f.read()
        assert 'Test log message' in log_content


def test_validate_settings_valid(valid_settings):
    """Test settings validation with valid settings."""
    try:
        validate_settings(valid_settings)
    except ValueError:
        pytest.fail("Validation failed for valid settings")


@pytest.mark.parametrize('key,value,expected_error', [
    (
        'transcription_mode',
        'invalid',
        "Invalid value for setting 'transcription_mode'"
    ),
    (
        'model_size',
        'huge',
        "Invalid value for setting 'model_size'"
    ),
    (
        'vad_sensitivity',
        0,
        "Invalid value for setting 'vad_sensitivity'"
    ),
    (
        'vad_sensitivity',
        4,
        "Invalid value for setting 'vad_sensitivity'"
    ),
    (
        'auto_detect_speech',
        'yes',
        "Error validating setting 'auto_detect_speech'"
    ),
    (
        'min_speech_duration',
        0,
        "Invalid value for setting 'min_speech_duration'"
    ),
    (
        'min_speech_duration',
        -1,
        "Invalid value for setting 'min_speech_duration'"
    ),
    (
        'speech_threshold',
        -0.1,
        "Invalid value for setting 'speech_threshold'"
    ),
    (
        'speech_threshold',
        1.1,
        "Invalid value for setting 'speech_threshold'"
    ),
    (
        'sample_rate',
        44100,
        "Invalid value for setting 'sample_rate'"
    ),
])
def test_validate_settings_invalid(valid_settings, key, value, expected_error):
    """Test settings validation with invalid settings."""
    settings = valid_settings.copy()
    settings[key] = value
    
    with pytest.raises(ValueError) as exc_info:
        validate_settings(settings)
    assert expected_error in str(exc_info.value)


def test_load_settings_default(temp_config_dir):
    """Test loading default settings when no config file exists."""
    settings = load_settings()
    assert settings['transcription_mode'] == 'continuous'
    assert settings['model_size'] == 'base'
    assert settings['vad_sensitivity'] == 3


def test_load_settings_custom(temp_config_dir, valid_settings):
    """Test loading custom settings from config file."""
    config_dir = temp_config_dir / '.config' / 'whisper-widget'
    config_dir.mkdir(parents=True)
    config_file = config_dir / 'config.json'
    
    # Modify some settings
    custom_settings = valid_settings.copy()
    custom_settings.update({
        'transcription_mode': 'clipboard',
        'model_size': 'small',
        'vad_sensitivity': 2
    })
    
    with open(config_file, 'w') as f:
        json.dump(custom_settings, f)
    
    settings = load_settings()
    assert settings['transcription_mode'] == 'clipboard'
    assert settings['model_size'] == 'small'
    assert settings['vad_sensitivity'] == 2


def test_load_settings_invalid_json(temp_config_dir):
    """Test loading settings with invalid JSON."""
    config_dir = temp_config_dir / '.config' / 'whisper-widget'
    config_dir.mkdir(parents=True)
    config_file = config_dir / 'config.json'
    
    with open(config_file, 'w') as f:
        f.write('invalid json')
    
    with pytest.raises(SettingsLoadError) as exc_info:
        load_settings()
    assert "Invalid JSON in settings file" in str(exc_info.value)


def test_load_settings_invalid_values(temp_config_dir, valid_settings):
    """Test loading settings with invalid values."""
    config_dir = temp_config_dir / '.config' / 'whisper-widget'
    config_dir.mkdir(parents=True)
    config_file = config_dir / 'config.json'
    
    invalid_settings = valid_settings.copy()
    invalid_settings['vad_sensitivity'] = 0
    
    with open(config_file, 'w') as f:
        json.dump(invalid_settings, f)
    
    with pytest.raises(SettingsLoadError) as exc_info:
        load_settings()
    assert "Invalid settings values" in str(exc_info.value)


def test_save_settings_valid(temp_config_dir, valid_settings):
    """Test saving valid settings."""
    save_settings(valid_settings)
    
    config_file = temp_config_dir / '.config' / 'whisper-widget' / 'config.json'
    assert config_file.exists()
    
    with open(config_file) as f:
        saved_settings = json.load(f)
    
    assert saved_settings == valid_settings


def test_save_settings_invalid(temp_config_dir, valid_settings):
    """Test saving invalid settings."""
    invalid_settings = valid_settings.copy()
    invalid_settings['vad_sensitivity'] = 0
    
    with pytest.raises(SettingsSaveError) as exc_info:
        save_settings(invalid_settings)
    assert "Invalid settings values" in str(exc_info.value)


def test_get_setting(valid_settings):
    """Test getting setting values."""
    assert get_setting(valid_settings, 'model_size') == 'base'
    assert get_setting(valid_settings, 'nonexistent') is None
    assert get_setting(valid_settings, 'nonexistent', 'default') == 'default' 