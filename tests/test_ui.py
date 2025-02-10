"""Tests for UI components."""

import pytest
from unittest.mock import MagicMock, patch
import gi

gi.require_version('Gtk', '4.0')
gi.require_version('WebKit2', '4.1')
gi.require_version('Gdk', '4.0')

from gi.repository import Gtk, Gio, GLib, WebKit2, Gdk
from whisper_widget.ui.window import WhisperWindow
from whisper_widget.ui.menu import (
    create_app_menu,
    _create_transcription_menu,
    _create_output_menu,
    _create_model_menu,
    _create_language_menu,
    _create_detection_menu
)


@pytest.fixture
def mock_gtk():
    """Mock GTK and related components."""
    with patch('gi.repository.Gtk') as mock_gtk, \
         patch('gi.repository.WebKit2') as mock_webkit, \
         patch('gi.repository.Gdk') as mock_gdk:
        
        # Create mock application
        mock_app = MagicMock()
        mock_gtk.Application.new.return_value = mock_app
        
        # Create mock box
        mock_box = MagicMock()
        mock_gtk.Box.new.return_value = mock_box
        
        # Create mock WebView
        mock_webview = MagicMock()
        mock_webkit.WebView.new.return_value = mock_webview
        
        # Create mock CSS provider
        mock_css = MagicMock()
        mock_gtk.CssProvider.new.return_value = mock_css
        
        # Create mock display
        mock_display = MagicMock()
        mock_gdk.Display.get_default.return_value = mock_display
        
        yield mock_gtk


@pytest.fixture
def mock_window(mock_gtk):
    """Create a mock window with callbacks."""
    mock_start = MagicMock()
    mock_stop = MagicMock()
    
    window = WhisperWindow(
        mock_gtk.Application(),
        on_recording_start=mock_start,
        on_recording_stop=mock_stop
    )
    
    return window


def test_window_initialization(mock_window):
    """Test window initialization."""
    assert mock_window.on_recording_start is not None
    assert mock_window.on_recording_stop is not None
    assert hasattr(mock_window, 'webview')
    assert hasattr(mock_window, 'box')


def test_window_transparency(mock_window, mock_gtk):
    """Test window transparency setup."""
    mock_gtk.CssProvider.new.assert_called_once()
    mock_gtk.StyleContext.add_provider_for_display.assert_called_once()


def test_window_events(mock_window):
    """Test window event handlers setup."""
    # Verify gesture controllers were added
    assert len(mock_window.list_controllers()) >= 2


def test_window_drag(mock_window):
    """Test window dragging functionality."""
    # Simulate drag begin
    mock_window._on_drag_begin(None, 100, 100)
    assert hasattr(mock_window, '_drag_start_pos')
    
    # Simulate drag update
    mock_window._on_drag_update(None, 50, 50)
    mock_window.move.assert_called_once()


def test_window_visualization(mock_window):
    """Test visualization state updates."""
    mock_window.update_visualization('recording')
    mock_window.webview.evaluate_javascript.assert_called_once()


def test_create_app_menu():
    """Test application menu creation."""
    mock_app = MagicMock()
    menu = create_app_menu(mock_app)
    assert isinstance(menu, Gio.Menu)


def test_transcription_menu():
    """Test transcription mode menu creation."""
    mock_app = MagicMock()
    menu = _create_transcription_menu(mock_app)
    assert isinstance(menu, Gio.Menu)
    assert mock_app.add_action.call_count == 2  # local and openai


def test_output_menu():
    """Test output mode menu creation."""
    mock_app = MagicMock()
    menu = _create_output_menu(mock_app)
    assert isinstance(menu, Gio.Menu)
    assert mock_app.add_action.call_count == 2  # continuous and clipboard


def test_model_menu():
    """Test model size menu creation."""
    mock_app = MagicMock()
    menu = _create_model_menu(mock_app)
    assert isinstance(menu, Gio.Menu)
    assert mock_app.add_action.call_count == 5  # tiny to large


def test_language_menu():
    """Test language menu creation."""
    mock_app = MagicMock()
    menu = _create_language_menu(mock_app)
    assert isinstance(menu, Gio.Menu)
    assert mock_app.add_action.call_count == 12  # number of languages


def test_detection_menu():
    """Test speech detection menu creation."""
    mock_app = MagicMock()
    menu = _create_detection_menu(mock_app)
    assert isinstance(menu, Gio.Menu)
    
    # Count submenu actions
    vad_actions = 3  # VAD levels
    start_actions = 4  # Speech start chunks
    silence_actions = 4  # Silence thresholds
    duration_actions = 9  # 3 settings x 3 values each
    
    total_actions = vad_actions + start_actions + silence_actions + duration_actions
    assert mock_app.add_action.call_count == total_actions 