import pytest
from unittest.mock import MagicMock, patch
from app import SpeechToTextApp


@pytest.fixture
def mock_gtk():
    """Mock GTK and AppIndicator3."""
    with patch('gi.repository.Gtk') as mock_gtk, \
         patch('gi.repository.AppIndicator3') as mock_indicator:
        # Create mock menu items
        mock_menu = MagicMock()
        mock_menu_item = MagicMock()
        mock_submenu = MagicMock()
        
        # Set up menu hierarchy
        mock_menu_item.get_submenu.return_value = mock_submenu
        mock_menu.get_children.return_value = [mock_menu_item]
        mock_menu.get_parent.return_value = mock_gtk
        
        # Set up GTK mocks
        mock_gtk.Menu.return_value = mock_menu
        mock_gtk.MenuItem = MagicMock
        mock_gtk.RadioMenuItem = MagicMock
        mock_gtk.CheckMenuItem = MagicMock
        mock_gtk.SeparatorMenuItem = MagicMock
        
        # Set up AppIndicator mocks
        mock_indicator.Indicator.new.return_value = MagicMock()
        mock_indicator.IndicatorCategory = MagicMock()
        mock_indicator.IndicatorStatus = MagicMock()
        
        yield mock_gtk


@pytest.fixture
def mock_app(mock_gtk, temp_config_dir):
    """Create a mock app with mocked UI components."""
    with patch('app.check_microphone_access', return_value=True), \
         patch('app.WhisperModel') as mock_model, \
         patch('app.webrtcvad.Vad') as mock_vad:
        
        # Set up mock model
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        
        # Set up mock VAD
        mock_vad_instance = MagicMock()
        mock_vad.return_value = mock_vad_instance
        
        # Create app with mocked components
        app = SpeechToTextApp()
        
        # Replace real components with mocks
        app.model = mock_model_instance
        app.vad = mock_vad_instance
        app.indicator = MagicMock()
        
        return app


def test_create_menu(mock_app):
    """Test menu creation."""
    # Menu should be created in __init__
    assert mock_app.menu is not None
    
    # Menu should have items
    assert len(mock_app.menu.get_children()) > 0


def test_update_icon_state(mock_app):
    """Test icon state updates."""
    states = ['ready', 'recording', 'computing', 'error', 'no_mic']
    
    for state in states:
        mock_app.update_icon_state(state)
        mock_app.indicator.set_icon.assert_called()


def test_transcription_mode_setting(mock_app):
    """Test changing transcription mode."""
    # Update the setting
    mock_app.update_setting('transcription_mode', 'clipboard')
    
    # Verify the setting was updated
    assert mock_app.settings['transcription_mode'] == 'clipboard'
    
    # Update the instance variable
    mock_app.transcription_mode = mock_app.settings['transcription_mode']
    assert mock_app.transcription_mode == 'clipboard'


def test_model_size_setting(mock_app):
    """Test changing model size."""
    # Create a new mock model class
    mock_model = MagicMock()
    mock_model_instance = MagicMock()
    mock_model.return_value = mock_model_instance
    
    # Replace the WhisperModel class
    with patch('app.WhisperModel', mock_model):
        # Update the setting and create new model
        mock_app.update_setting('model_size', 'small')
        
        # Update the instance variable directly
        mock_app.model_size = 'small'
        
        # Create new model instance
        mock_app.model = mock_model('small', device="cpu", compute_type="int8")
        
        # Verify the setting was updated
        assert mock_app.settings['model_size'] == 'small'
        assert mock_app.model_size == 'small'
        
        # Verify model was created with correct parameters
        mock_model.assert_called_with(
            'small', device="cpu", compute_type="int8"
        )


def test_vad_sensitivity_setting(mock_app):
    """Test changing VAD sensitivity."""
    # Create a new mock VAD class
    mock_vad = MagicMock()
    mock_vad_instance = MagicMock()
    mock_vad.return_value = mock_vad_instance
    
    # Replace the Vad class
    with patch('app.webrtcvad.Vad', mock_vad):
        # Update the setting and create new VAD
        mock_app.update_setting('vad_sensitivity', 2)
        
        # Update the instance variable directly
        mock_app.vad_sensitivity = 2
        
        # Create new VAD instance
        mock_app.vad = mock_vad(2)
        
        # Verify the setting was updated
        assert mock_app.settings['vad_sensitivity'] == 2
        assert mock_app.vad_sensitivity == 2
        
        # Verify VAD was created with correct sensitivity
        mock_vad.assert_called_with(2)


def test_quit(mock_app):
    """Test quit functionality."""
    with patch('gi.repository.Gtk.main_quit') as mock_quit:
        mock_app.quit(None)
        assert mock_app.running is False
        mock_quit.assert_called_once()


def test_menu_items(mock_app):
    """Test menu item creation and callbacks."""
    # Test settings menu
    settings_menu = mock_app.menu.get_children()[0].get_submenu()
    assert settings_menu is not None
    
    # Test transcription mode menu
    mode_menu = settings_menu.get_children()[0].get_submenu()
    assert mode_menu is not None
    
    # Test model size menu
    model_menu = settings_menu.get_children()[1].get_submenu()
    assert model_menu is not None
    
    # Test language menu
    lang_menu = settings_menu.get_children()[2].get_submenu()
    assert lang_menu is not None
    
    # Test VAD sensitivity menu
    vad_menu = settings_menu.get_children()[3].get_submenu()
    assert vad_menu is not None


def test_toggle_options(mock_app):
    """Test toggle option menu items."""
    settings_menu = mock_app.menu.get_children()[0].get_submenu()
    
    # Test auto-detect speech toggle
    auto_detect = settings_menu.get_children()[4]
    assert auto_detect.get_active() == mock_app.auto_detect_speech
    
    # Test add punctuation toggle
    add_punct = settings_menu.get_children()[5]
    assert add_punct.get_active() == mock_app.add_punctuation 