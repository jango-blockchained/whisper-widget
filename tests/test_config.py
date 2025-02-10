"""Tests for configuration management module."""

import json
from pathlib import Path
import pytest
from whisper_widget.config import (
    ConfigVersionError,
    ConfigBackupError,
    get_config_version,
    version_to_tuple,
    needs_migration,
    migrate_settings,
    backup_settings,
    restore_settings,
    list_backups,
    CURRENT_VERSION
)


@pytest.fixture
def old_settings():
    """Return a dictionary of old (pre-1.0.0) settings."""
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
        'silence_threshold': 10,  # Old name
        'speech_start_chunks': 2,
        'noise_reduce_threshold': 0.1,
        'sample_rate': 16000,
    }


def test_get_config_version_default():
    """Test getting config version with no version set."""
    settings = {'test': 'value'}
    assert get_config_version(settings) == "0.0.0"


def test_get_config_version_set():
    """Test getting config version when set."""
    settings = {'config_version': "1.0.0"}
    assert get_config_version(settings) == "1.0.0"


def test_version_to_tuple():
    """Test version string to tuple conversion."""
    assert version_to_tuple("1.0.0") == (1, 0, 0)
    assert version_to_tuple("2.3.4") == (2, 3, 4)
    assert version_to_tuple("0.0.0") == (0, 0, 0)


def test_needs_migration():
    """Test migration check."""
    assert needs_migration({'config_version': "0.0.0"})
    assert needs_migration({'config_version': "0.9.9"})
    assert not needs_migration({'config_version': CURRENT_VERSION})
    assert not needs_migration({'config_version': "2.0.0"})


def test_migrate_settings_old_to_current(old_settings):
    """Test migrating old settings to current version."""
    migrated = migrate_settings(old_settings)
    
    # Check version updated
    assert migrated['config_version'] == CURRENT_VERSION
    
    # Check renamed fields
    assert 'silence_threshold' not in migrated
    assert 'silence_db_threshold' in migrated
    assert migrated['silence_db_threshold'] == 10
    
    # Check new fields
    assert migrated['backup_enabled'] is True
    assert migrated['max_backups'] == 5
    
    # Check unchanged fields
    assert migrated['transcription_mode'] == old_settings['transcription_mode']
    assert migrated['model_size'] == old_settings['model_size']
    assert migrated['vad_sensitivity'] == old_settings['vad_sensitivity']


def test_migrate_settings_invalid():
    """Test migrating invalid settings."""
    invalid_settings = {
        'config_version': "0.0.0",
        'vad_sensitivity': 0  # Invalid value
    }
    
    with pytest.raises(ConfigVersionError) as exc_info:
        migrate_settings(invalid_settings)
    assert "Failed to migrate settings" in str(exc_info.value)


def test_backup_settings(temp_config_dir, old_settings):
    """Test backing up settings."""
    # Create initial config
    config_dir = temp_config_dir / '.config' / 'whisper-widget'
    config_dir.mkdir(parents=True)
    with open(config_dir / 'config.json', 'w') as f:
        json.dump(old_settings, f)
    
    # Create backup
    backup_file = backup_settings()
    assert backup_file is not None
    assert backup_file.exists()
    assert backup_file.parent == config_dir / 'backups'
    
    # Check backup contents
    with open(backup_file) as f:
        backup_data = json.load(f)
    assert backup_data == old_settings


def test_backup_settings_disabled(temp_config_dir, old_settings):
    """Test backup when disabled."""
    # Create initial config with backups disabled
    config_dir = temp_config_dir / '.config' / 'whisper-widget'
    config_dir.mkdir(parents=True)
    settings = old_settings.copy()
    settings['backup_enabled'] = False
    with open(config_dir / 'config.json', 'w') as f:
        json.dump(settings, f)
    
    # Attempt backup
    backup_file = backup_settings()
    assert backup_file is None
    assert not (config_dir / 'backups').exists()


def test_backup_cleanup(temp_config_dir, old_settings):
    """Test cleanup of old backups."""
    # Create initial config
    config_dir = temp_config_dir / '.config' / 'whisper-widget'
    backup_dir = config_dir / 'backups'
    config_dir.mkdir(parents=True)
    backup_dir.mkdir()
    
    # Create initial config with max_backups=2
    settings = old_settings.copy()
    settings['max_backups'] = 2
    with open(config_dir / 'config.json', 'w') as f:
        json.dump(settings, f)
    
    # Create some test backup files with different timestamps
    timestamps = ['20240101_120000', '20240101_120001', '20240101_120002']
    backup_files = []
    for ts in timestamps:
        backup_file = backup_dir / f'config_{ts}.json'
        backup_files.append(backup_file)
        with open(backup_file, 'w') as f:
            json.dump(settings, f)
    
    # Create new backup (should trigger cleanup)
    new_backup = backup_settings()
    assert new_backup is not None
    
    # Check only newest backups remain
    remaining = list_backups()
    assert len(remaining) == 2
    assert new_backup in remaining
    assert backup_files[-1] in remaining
    assert backup_files[0] not in remaining


def test_restore_settings(temp_config_dir, old_settings):
    """Test restoring settings from backup."""
    # Create backup file
    config_dir = temp_config_dir / '.config' / 'whisper-widget'
    backup_dir = config_dir / 'backups'
    backup_dir.mkdir(parents=True)
    backup_file = backup_dir / 'config_test.json'
    with open(backup_file, 'w') as f:
        json.dump(old_settings, f)
    
    # Restore settings
    restored = restore_settings(backup_file)
    
    # Check restored settings
    assert restored['config_version'] == CURRENT_VERSION
    assert 'silence_db_threshold' in restored
    assert (
        restored['silence_db_threshold'] == old_settings['silence_threshold']
    )


def test_restore_settings_nonexistent():
    """Test restoring from nonexistent backup."""
    with pytest.raises(ConfigBackupError) as exc_info:
        restore_settings(Path('nonexistent.json'))
    assert "Backup file not found" in str(exc_info.value)


def test_restore_settings_invalid(temp_config_dir):
    """Test restoring invalid settings."""
    # Create invalid backup file
    config_dir = temp_config_dir / '.config' / 'whisper-widget'
    backup_dir = config_dir / 'backups'
    backup_dir.mkdir(parents=True)
    backup_file = backup_dir / 'config_test.json'
    with open(backup_file, 'w') as f:
        f.write('invalid json')
    
    with pytest.raises(ConfigBackupError) as exc_info:
        restore_settings(backup_file)
    assert "Failed to restore settings" in str(exc_info.value)


def test_list_backups(temp_config_dir, old_settings):
    """Test listing backup files."""
    # Create backup directory with test files
    backup_dir = temp_config_dir / '.config' / 'whisper-widget' / 'backups'
    backup_dir.mkdir(parents=True)
    
    # Create test backup files
    timestamps = ['20240101_120000', '20240101_120001']
    backup_files = []
    for ts in timestamps:
        backup_file = backup_dir / f'config_{ts}.json'
        backup_files.append(backup_file)
        with open(backup_file, 'w') as f:
            json.dump(old_settings, f)
    
    # List backups
    backups = list_backups()
    assert len(backups) == 2
    assert backups == sorted(backup_files, reverse=True)


def test_list_backups_empty(temp_config_dir):
    """Test listing backups with no backup directory."""
    assert list_backups() == [] 