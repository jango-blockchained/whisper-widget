"""Speech-to-text application using GTK 3."""

# Set GTK version before ANY other imports
import gi
gi.require_version('Gtk', '3.0')
gi.require_version('Gdk', '3.0')
gi.require_version('WebKit2', '4.1')

# Check if system tray is available
HAS_INDICATOR = False
try:
    gi.require_version('AyatanaAppIndicator3', '0.1')
    from gi.repository import AyatanaAppIndicator3
    HAS_INDICATOR = True
except (ValueError, ImportError):
    import logging
    logging.warning("System tray support not available (requires GTK 3.0)")

from .app import SpeechToTextApp

__version__ = "0.1.0"
__all__ = ["SpeechToTextApp"]
