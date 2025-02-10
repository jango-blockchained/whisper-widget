"""Advanced configuration management module."""

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from whisper_widget.settings import (
    SettingsError,
    load_settings,
    save_settings,
    validate_settings
)


logger = logging.getLogger(__name__)


class ConfigError(SettingsError):
    """Base exception for configuration-related errors."""
    pass


class ConfigVersionError(ConfigError):
    """Raised when there's a version mismatch in configuration."""
    pass


class ConfigBackupError(ConfigError):
    """Raised when backup/restore operations fail."""
    pass


CURRENT_VERSION = "1.0.0"


def get_config_version(settings: Dict[str, Any]) -> str:
    """Get the version of the configuration.
    
    Args:
        settings: Dictionary of settings.
    
    Returns:
        Version string or "0.0.0" if not found.
    """
    return settings.get('config_version', "0.0.0")


def version_to_tuple(version: str) -> Tuple[int, ...]:
    """Convert version string to tuple for comparison.
    
    Args:
        version: Version string in format "x.y.z".
    
    Returns:
        Tuple of version numbers.
    """
    return tuple(int(x) for x in version.split('.'))


def needs_migration(settings: Dict[str, Any]) -> bool:
    """Check if settings need migration.
    
    Args:
        settings: Dictionary of settings.
    
    Returns:
        True if migration is needed, False otherwise.
    """
    current = version_to_tuple(CURRENT_VERSION)
    config = version_to_tuple(get_config_version(settings))
    return config < current


def migrate_settings(settings: Dict[str, Any]) -> Dict[str, Any]:
    """Migrate settings to current version.
    
    Args:
        settings: Dictionary of settings to migrate.
    
    Returns:
        Migrated settings dictionary.
    
    Raises:
        ConfigVersionError: If migration fails.
    """
    current_version = get_config_version(settings)
    
    try:
        # Version 0.0.0 to 1.0.0
        if version_to_tuple(current_version) < version_to_tuple("1.0.0"):
            settings = _migrate_to_1_0_0(settings)
        
        # Add more version migrations here
        
        settings['config_version'] = CURRENT_VERSION
        validate_settings(settings)
        return settings
    
    except Exception as e:
        msg = f"Failed to migrate settings from {current_version}: {e}"
        logger.error(msg)
        raise ConfigVersionError(msg)


def _migrate_to_1_0_0(settings: Dict[str, Any]) -> Dict[str, Any]:
    """Migrate settings from 0.0.0 to 1.0.0.
    
    Version 1.0.0 changes:
    - Adds config_version field
    - Renames 'silence_threshold' to 'silence_db_threshold'
    - Adds 'backup_enabled' and 'max_backups' settings
    
    Args:
        settings: Settings dictionary to migrate.
    
    Returns:
        Migrated settings dictionary.
    """
    migrated = settings.copy()
    
    # Rename silence_threshold if it exists
    if 'silence_threshold' in migrated:
        migrated['silence_db_threshold'] = migrated.pop('silence_threshold')
    
    # Add new settings with defaults
    migrated.setdefault('backup_enabled', True)
    migrated.setdefault('max_backups', 5)
    
    return migrated


def backup_settings() -> Optional[Path]:
    """Create a backup of current settings.
    
    Returns:
        Path to backup file if successful, None if backup is disabled.
    
    Raises:
        ConfigBackupError: If backup fails.
    """
    try:
        settings = load_settings()
        if not settings.get('backup_enabled', True):
            logger.info("Settings backup is disabled")
            return None
        
        config_dir = Path.home() / '.config' / 'whisper-widget'
        backup_dir = config_dir / 'backups'
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Create backup filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_file = backup_dir / f'config_{timestamp}.json'
        
        # Copy current config to backup
        shutil.copy2(config_dir / 'config.json', backup_file)
        
        # Clean up old backups
        _cleanup_old_backups(backup_dir, settings.get('max_backups', 5))
        
        logger.info("Settings backed up to %s", backup_file)
        return backup_file
    
    except Exception as e:
        msg = f"Failed to backup settings: {e}"
        logger.error(msg)
        raise ConfigBackupError(msg)


def restore_settings(backup_file: Path) -> Dict[str, Any]:
    """Restore settings from a backup file.
    
    Args:
        backup_file: Path to backup file to restore from.
    
    Returns:
        Restored settings dictionary.
    
    Raises:
        ConfigBackupError: If restore fails.
    """
    try:
        if not backup_file.exists():
            msg = f"Backup file not found: {backup_file}"
            logger.error(msg)
            raise ConfigBackupError(msg)
        
        # Read backup file
        with open(backup_file) as f:
            settings = json.load(f)
        
        # Validate and migrate if needed
        validate_settings(settings)
        if needs_migration(settings):
            settings = migrate_settings(settings)
        
        # Save restored settings
        save_settings(settings)
        
        logger.info("Settings restored from %s", backup_file)
        return settings
    
    except Exception as e:
        msg = f"Failed to restore settings from {backup_file}: {e}"
        logger.error(msg)
        raise ConfigBackupError(msg)


def list_backups() -> List[Path]:
    """List available backup files.
    
    Returns:
        List of paths to backup files, sorted by date (newest first).
    """
    backup_dir = Path.home() / '.config' / 'whisper-widget' / 'backups'
    if not backup_dir.exists():
        return []
    
    backups = list(backup_dir.glob('config_*.json'))
    return sorted(backups, reverse=True)


def _cleanup_old_backups(backup_dir: Path, max_backups: int) -> None:
    """Remove old backup files, keeping only the specified number.
    
    Args:
        backup_dir: Directory containing backup files.
        max_backups: Maximum number of backups to keep.
    """
    backups = list_backups()
    if len(backups) > max_backups:
        for backup in backups[max_backups:]:
            try:
                backup.unlink()
                logger.debug("Removed old backup: %s", backup)
            except Exception as e:
                logger.warning("Failed to remove old backup %s: %s", backup, e) 