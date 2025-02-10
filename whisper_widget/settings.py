"""Settings management module."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

# Set up logging
logger = logging.getLogger(__name__)

class SettingsError(Exception):
    """Base exception for settings-related errors."""
    pass

class SettingsLoadError(SettingsError):
    """Raised when settings cannot be loaded."""
    pass

class SettingsSaveError(SettingsError):
    """Raised when settings cannot be saved."""
    pass

def setup_logging(level: int = logging.INFO) -> None:
    """Set up logging configuration.
    
    Args:
        level: The logging level to use. Defaults to INFO.
    """
    config_dir = Path.home() / '.config' / 'whisper-widget'
    log_file = config_dir / 'app.log'
    config_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )

def validate_settings(settings: Dict[str, Any]) -> None:
    """Validate settings values.
    
    Args:
        settings: Dictionary of settings to validate.
    
    Raises:
        ValueError: If any setting has an invalid value.
    """
    model_sizes = ['tiny', 'base', 'small', 'medium', 'large', 'large-v3', 'large-v2', 'large-v3-turbo']
    trans_modes = ['local', 'openai']  # How transcription is performed
    output_modes = ['continuous', 'clipboard']  # How text is output
    
    validators = {
        'transcription_mode': lambda x: x in trans_modes,
        'output_mode': lambda x: x in output_modes,
        'model_size': lambda x: x in model_sizes,
        'vad_sensitivity': lambda x: isinstance(x, int) and 1 <= x <= 3,
        'auto_detect_speech': lambda x: isinstance(x, bool),
        'add_punctuation': lambda x: isinstance(x, bool),
        'min_speech_duration': lambda x: isinstance(x, (int, float)) and x > 0,
        'max_silence_duration': lambda x: isinstance(x, (int, float)) and x > 0,
        'min_audio_length': lambda x: isinstance(x, (int, float)) and x > 0,
        'speech_threshold': lambda x: isinstance(x, (int, float)) and 0 <= x <= 1,
        'silence_threshold': lambda x: isinstance(x, (int, float)) and x > 0,
        'speech_start_chunks': lambda x: isinstance(x, int) and x > 0,
        'noise_reduce_threshold': (
            lambda x: isinstance(x, (int, float)) and 0 <= x <= 1
        ),
        'sample_rate': lambda x: x == 16000  # Fixed for Whisper
    }
    
    for key, validator in validators.items():
        if key in settings:
            try:
                if not validator(settings[key]):
                    msg = f"Invalid value for setting '{key}': {settings[key]}"
                    raise ValueError(msg)
            except Exception as e:
                msg = f"Error validating setting '{key}': {e}"
                raise ValueError(msg)

def load_settings() -> Dict[str, Any]:
    """Load settings from config file.
    
    Returns:
        Dictionary containing the settings.
    
    Raises:
        SettingsLoadError: If settings cannot be loaded.
    """
    config_dir = Path.home() / '.config' / 'whisper-widget'
    config_file = config_dir / 'config.json'
    default_settings: Dict[str, Any] = {
        'transcription_mode': 'local',  # local or openai
        'output_mode': 'continuous',  # continuous or clipboard
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
    
    if config_file.exists():
        try:
            with open(config_file) as f:
                loaded_settings = json.load(f)
                settings = {**default_settings, **loaded_settings}
                validate_settings(settings)
                logger.info("Settings loaded successfully")
                return settings
        except json.JSONDecodeError as e:
            logger.error("Failed to parse settings file: %s", e)
            msg = f"Invalid JSON in settings file: {e}"
            raise SettingsLoadError(msg)
        except ValueError as e:
            logger.error("Invalid settings values: %s", e)
            msg = f"Invalid settings values: {e}"
            raise SettingsLoadError(msg)
        except Exception as e:
            logger.error("Unexpected error loading settings: %s", e)
            msg = f"Failed to load settings: {e}"
            raise SettingsLoadError(msg)
    
    logger.info("Using default settings")
    return default_settings

def save_settings(settings: Dict[str, Any]) -> None:
    """Save settings to config file.
    
    Args:
        settings: Dictionary of settings to save.
    
    Raises:
        SettingsSaveError: If settings cannot be saved.
    """
    config_dir = Path.home() / '.config' / 'whisper-widget'
    config_file = config_dir / 'config.json'
    
    try:
        validate_settings(settings)
        config_dir.mkdir(parents=True, exist_ok=True)
        
        with open(config_file, 'w') as f:
            json.dump(settings, f, indent=4)
        logger.info("Settings saved successfully")
    except ValueError as e:
        logger.error("Invalid settings values: %s", e)
        msg = f"Invalid settings values: {e}"
        raise SettingsSaveError(msg)
    except Exception as e:
        logger.error("Failed to save settings: %s", e)
        msg = f"Failed to save settings: {e}"
        raise SettingsSaveError(msg)

def get_setting(settings: Dict[str, Any], key: str, default: Optional[Any] = None) -> Any:
    """Safely get a setting value with optional default.
    
    Args:
        settings: Dictionary of settings.
        key: Setting key to retrieve.
        default: Default value if setting doesn't exist.
    
    Returns:
        The setting value or default if not found.
    """
    return settings.get(key, default) 