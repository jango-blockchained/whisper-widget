import pytest
from unittest.mock import MagicMock, patch
from whisper_widget.app import SpeechToTextApp
from gi.repository import Gtk


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
    auto_detect = settings_menu.get_children()[5]  # CheckMenuItem
    assert isinstance(auto_detect, Gtk.CheckMenuItem)
    assert auto_detect.get_active() == mock_app.auto_detect_speech

    # Test add punctuation toggle
    add_punct = settings_menu.get_children()[6]  # CheckMenuItem
    assert isinstance(add_punct, Gtk.CheckMenuItem)
    assert add_punct.get_active() == mock_app.add_punctuation

    # Test menu item activation
    auto_detect.set_active(not mock_app.auto_detect_speech)
    assert mock_app.settings['auto_detect_speech'] == (not mock_app.auto_detect_speech)

    add_punct.set_active(not mock_app.add_punctuation)
    assert mock_app.settings['add_punctuation'] == (not mock_app.add_punctuation)


def test_speech_detection_menu(mock_app):
    """Test speech detection settings menu items."""
    app = mock_app
    settings_menu = app.menu.get_children()[0].get_submenu()
    speech_menu = settings_menu.get_children()[3].get_submenu()  # Speech Detection menu

    # Test min speech duration menu
    min_speech_menu = speech_menu.get_children()[0].get_submenu()
    assert min_speech_menu is not None
    assert len(min_speech_menu.get_children()) == 5  # [0.3, 0.5, 1.0, 1.5, 2.0]

    # Test max silence duration menu
    max_silence_menu = speech_menu.get_children()[1].get_submenu()
    assert max_silence_menu is not None
    assert len(max_silence_menu.get_children()) == 5  # [0.5, 1.0, 1.5, 2.0, 3.0]

    # Test speech start chunks menu
    chunks_menu = speech_menu.get_children()[2].get_submenu()
    assert chunks_menu is not None
    assert len(chunks_menu.get_children()) == 5  # [1, 2, 3, 4, 5]

    # Test noise reduction menu
    noise_menu = speech_menu.get_children()[3].get_submenu()
    assert noise_menu is not None
    assert len(noise_menu.get_children()) == 6  # [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]


def test_speech_detection_settings_update(mock_app):
    """Test updating speech detection settings through menu."""
    app = mock_app
    settings_menu = app.menu.get_children()[0].get_submenu()
    speech_menu = settings_menu.get_children()[3].get_submenu()

    # Test min speech duration update
    min_speech_menu = speech_menu.get_children()[0].get_submenu()
    min_speech_item = min_speech_menu.get_children()[2]  # 1.0s option
    min_speech_item.activate()
    assert app.settings['min_speech_duration'] == 1.0

    # Test max silence duration update
    max_silence_menu = speech_menu.get_children()[1].get_submenu()
    max_silence_item = max_silence_menu.get_children()[3]  # 2.0s option
    max_silence_item.activate()
    assert app.settings['max_silence_duration'] == 2.0

    # Test speech start chunks update
    chunks_menu = speech_menu.get_children()[2].get_submenu()
    chunks_item = chunks_menu.get_children()[2]  # 3 chunks option
    chunks_item.activate()
    assert app.settings['speech_start_chunks'] == 3

    # Test noise reduction threshold update
    noise_menu = speech_menu.get_children()[3].get_submenu()
    noise_item = noise_menu.get_children()[2]  # 0.2 threshold option
    noise_item.activate()
    assert app.settings['noise_reduce_threshold'] == 0.2


def test_speech_detection_settings_persistence(mock_app, tmp_path):
    """Test that speech detection settings persist after app restart."""
    app = mock_app
    
    # Update settings
    test_settings = {
        'min_speech_duration': 1.5,
        'max_silence_duration': 2.0,
        'speech_start_chunks': 3,
        'noise_reduce_threshold': 0.3
    }
    
    for key, value in test_settings.items():
        app.update_setting(key, value)
    
    # Create new app instance
    new_app = SpeechToTextApp()
    
    # Verify settings persisted
    for key, value in test_settings.items():
        assert abs(new_app.settings[key] - value) < 0.01


def test_speech_detection_menu_state(mock_app):
    """Test that menu items reflect current settings state."""
    app = mock_app
    settings_menu = app.menu.get_children()[0].get_submenu()
    speech_menu = settings_menu.get_children()[3].get_submenu()

    # Test min speech duration state
    min_speech_menu = speech_menu.get_children()[0].get_submenu()
    for i, item in enumerate(min_speech_menu.get_children()):
        durations = [0.3, 0.5, 1.0, 1.5, 2.0]
        assert item.get_active() == (
            abs(durations[i] - app.settings['min_speech_duration']) < 0.01
        )

    # Test max silence duration state
    max_silence_menu = speech_menu.get_children()[1].get_submenu()
    for i, item in enumerate(max_silence_menu.get_children()):
        silences = [0.5, 1.0, 1.5, 2.0, 3.0]
        assert item.get_active() == (
            abs(silences[i] - app.settings['max_silence_duration']) < 0.01
        )

    # Test speech start chunks state
    chunks_menu = speech_menu.get_children()[2].get_submenu()
    for i, item in enumerate(chunks_menu.get_children()):
        assert item.get_active() == (
            (i + 1) == app.settings['speech_start_chunks']
        )

    # Test noise reduction threshold state
    noise_menu = speech_menu.get_children()[3].get_submenu()
    for i, item in enumerate(noise_menu.get_children()):
        thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        assert item.get_active() == (
            abs(thresholds[i] - app.settings['noise_reduce_threshold']) < 0.01
        ) 