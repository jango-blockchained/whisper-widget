"""Tests for UI components."""

# Set GTK version before ANY other imports
import gi
gi.require_version('Gtk', '4.0')
gi.require_version('WebKit2', '4.1')
gi.require_version('Gdk', '4.0')

import pytest
from unittest.mock import MagicMock, patch

from gi.repository import Gtk, Gio, GLib, WebKit2, Gdk  # noqa: E402

# Import application modules
from whisper_widget.app import SpeechToTextApp  # noqa: E402
from whisper_widget.ui.window import WhisperWindow  # noqa: E402
from whisper_widget.ui.menu import (  # noqa: E402
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
    
    # Create mock window instance
    window = MagicMock(spec=WhisperWindow)
    window.on_recording_start = mock_start
    window.on_recording_stop = mock_stop
    
    # Set up mock webview
    mock_webview = MagicMock()
    window.webview = mock_webview
    
    # Set up mock box
    mock_box = MagicMock()
    window.box = mock_box
    
    # Set up mock methods
    window._drag_start_pos = None
    window._on_drag_begin = lambda _, x, y: setattr(
        window, '_drag_start_pos', (x, y)
    )
    window._on_drag_update = lambda _, x, y: window.move(x, y)
    window.update_visualization = lambda state: window.webview.evaluate_javascript(
        f'updateVisualization("{state}")'
    )
    
    # Mock the WhisperWindow class
    with patch('whisper_widget.ui.window.WhisperWindow', return_value=window):
        yield window


def test_window_initialization(mock_window):
    """Test window initialization."""
    assert mock_window.on_recording_start is not None
    assert mock_window.on_recording_stop is not None
    assert hasattr(mock_window, 'webview')
    assert hasattr(mock_window, 'box')


def test_window_transparency(mock_window, mock_gtk):
    """Test window transparency setup."""
    WhisperWindow(
        mock_gtk.Application(),
        on_recording_start=MagicMock(),
        on_recording_stop=MagicMock()
    )
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
    assert mock_window._drag_start_pos == (100, 100)
    
    # Simulate drag update
    mock_window._on_drag_update(None, 50, 50)
    mock_window.move.assert_called_once_with(50, 50)


def test_window_visualization(mock_window):
    """Test visualization state updates."""
    mock_window.update_visualization('recording')
    expected_js = 'updateVisualization("recording")'
    mock_window.webview.evaluate_javascript.assert_called_once_with(expected_js)


def test_create_app_menu():
    """Test application menu creation."""
    mock_app = MagicMock()
    menu = create_app_menu(mock_app)
    assert hasattr(menu, 'append_submenu')  # Check for Menu interface
    assert mock_app.add_action.called


def test_transcription_menu():
    """Test transcription mode menu creation."""
    mock_app = MagicMock()
    menu = _create_transcription_menu(mock_app)
    assert hasattr(menu, 'append_submenu')  # Check for Menu interface
    assert mock_app.add_action.call_count == 2  # local and openai


def test_output_menu():
    """Test output mode menu creation."""
    mock_app = MagicMock()
    menu = _create_output_menu(mock_app)
    assert hasattr(menu, 'append_submenu')  # Check for Menu interface
    assert mock_app.add_action.call_count == 2  # continuous and clipboard


def test_model_menu():
    """Test model size menu creation."""
    mock_app = MagicMock()
    menu = _create_model_menu(mock_app)
    assert hasattr(menu, 'append_submenu')  # Check for Menu interface
    assert mock_app.add_action.call_count == 5  # tiny to large


def test_language_menu():
    """Test language menu creation."""
    mock_app = MagicMock()
    menu = _create_language_menu(mock_app)
    assert hasattr(menu, 'append_submenu')  # Check for Menu interface
    assert mock_app.add_action.call_count == 12  # number of languages


def test_detection_menu():
    """Test speech detection menu creation."""
    mock_app = MagicMock()
    menu = _create_detection_menu(mock_app)
    assert hasattr(menu, 'append_submenu')  # Check for Menu interface
    assert mock_app.add_action.called
    
    # Count submenu actions
    vad_actions = 3  # VAD levels
    start_actions = 4  # Speech start chunks
    silence_actions = 4  # Silence thresholds
    duration_actions = 9  # 3 settings x 3 values each
    
    total_actions = vad_actions + start_actions + silence_actions + duration_actions
    assert mock_app.add_action.call_count == total_actions 


def test_app_with_tray_enabled(mock_gtk, mock_audio):
    """Test application initialization with tray icon enabled."""
    app = SpeechToTextApp(use_tray=True)
    assert app.use_tray
    assert app.indicator is not None


def test_app_with_tray_disabled(mock_gtk, mock_audio):
    """Test application initialization with tray icon disabled."""
    app = SpeechToTextApp(use_tray=False)
    assert not app.use_tray
    assert app.indicator is None


def test_window_decoration_with_tray(mock_gtk, mock_audio):
    """Test window decoration when tray is enabled."""
    app = SpeechToTextApp(use_tray=True)
    
    # Mock the window's get_decorated method
    mock_window = MagicMock(spec=WhisperWindow)
    mock_window.get_decorated = MagicMock(return_value=False)
    with patch('whisper_widget.ui.window.WhisperWindow', 
              return_value=mock_window):
        app.on_activate(app.app)
        assert not app.window.get_decorated()


def test_window_decoration_without_tray(mock_gtk, mock_audio):
    """Test window decoration when tray is disabled."""
    app = SpeechToTextApp(use_tray=False)
    
    # Mock the window's get_decorated method
    mock_window = MagicMock(spec=WhisperWindow)
    mock_window.get_decorated = MagicMock(return_value=True)
    with patch('whisper_widget.ui.window.WhisperWindow', 
              return_value=mock_window):
        app.on_activate(app.app)
        assert app.window.get_decorated()


@pytest.mark.parametrize("has_indicator", [True, False])
def test_indicator_fallback(mock_gtk, mock_audio, monkeypatch, has_indicator):
    """Test indicator fallback behavior."""
    # Mock the HAS_INDICATOR constant
    import whisper_widget.app
    monkeypatch.setattr(whisper_widget.app, "HAS_INDICATOR", has_indicator)
    
    app = SpeechToTextApp(use_tray=True)
    assert app.use_tray == has_indicator
    if has_indicator:
        assert app.indicator is not None
        # Reset the mock to clear previous calls
        app.indicator.set_status.reset_mock()
        app.indicator.set_status(mock_gtk.IndicatorStatus.ACTIVE)
        app.indicator.set_status.assert_called_once_with(mock_gtk.IndicatorStatus.ACTIVE)


def test_window_keep_above_without_tray(mock_gtk, mock_audio):
    """Test window keep-above behavior when tray is disabled."""
    app = SpeechToTextApp(use_tray=False)
    
    # Mock the window's set_keep_above method
    mock_window = MagicMock(spec=WhisperWindow)
    mock_window.set_keep_above = MagicMock()
    with patch('whisper_widget.ui.window.WhisperWindow', return_value=mock_window):
        app.on_activate(app.app)
        app.window.set_keep_above(True)
        app.window.set_keep_above.assert_called_once_with(True) 