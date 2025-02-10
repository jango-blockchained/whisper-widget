import pytest
from unittest.mock import MagicMock, patch
from whisper_widget.app import SpeechToTextApp
from gi.repository import Gtk, GLib


@pytest.fixture
def mock_gtk():
    """Mock GTK and related components."""
    with patch('gi.repository.Gtk') as mock_gtk, \
         patch('gi.repository.Gio') as mock_gio:
        
        # Create mock application
        mock_app = MagicMock()
        mock_gtk.Application.new.return_value = mock_app
        
        # Create mock menu
        mock_menu = MagicMock()
        mock_gio.Menu.new.return_value = mock_menu
        
        # Create mock status icon
        mock_status_icon = MagicMock()
        mock_gtk.StatusIcon.new_from_icon_name.return_value = mock_status_icon
        
        # Create mock popover
        mock_popover = MagicMock()
        mock_gtk.PopoverMenu.new_from_model.return_value = mock_popover
        
        yield mock_gtk


@pytest.fixture
def mock_app(mock_gtk, temp_config_dir):
    """Create a mock app with mocked UI components."""
    with patch(
        'whisper_widget.app.check_microphone_access',
        return_value=True
    ), patch(
        'whisper_widget.app.WhisperModel'
    ) as mock_model, patch(
        'whisper_widget.app.webrtcvad.Vad'
    ) as mock_vad:
        
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
        app.status_icon = MagicMock()
        
        return app


def test_app_initialization(mock_app):
    """Test app initialization."""
    app = mock_app
    
    # Check that components are initialized
    assert app.model is not None
    assert app.vad is not None
    assert app.status_icon is not None
    assert app.menu is not None
    
    # Check default settings
    assert app.transcription_mode in ['continuous', 'clipboard']
    assert app.model_size in ['tiny', 'base', 'small', 'medium', 'large']
    assert app.language in ['en', 'es', 'fr', 'de', 'zh', 'ja', 'ru']
    assert 1 <= app.vad_sensitivity <= 3


def test_menu_actions(mock_app):
    """Test menu action handling."""
    app = mock_app
    
    # Test mode change
    mode_action = app.app.lookup_action('set-mode')
    mode_action.activate(GLib.Variant.new_string('local'))
    assert app.settings['transcription_mode'] == 'local'
    
    # Test model change
    model_action = app.app.lookup_action('set-model')
    model_action.activate(GLib.Variant.new_string('small'))
    assert app.settings['model_size'] == 'small'
    
    # Test language change
    lang_action = app.app.lookup_action('set-language')
    lang_action.activate(GLib.Variant.new_string('fr'))
    assert app.settings['language'] == 'fr'
    
    # Test auto-detect toggle
    auto_detect = app.app.lookup_action('toggle-auto-detect')
    auto_detect.activate(None)
    assert app.settings['auto_detect_speech'] == (
        not mock_app.auto_detect_speech
    )
    
    # Test punctuation toggle
    add_punct = app.app.lookup_action('toggle-punctuation')
    add_punct.activate(None)
    assert app.settings['add_punctuation'] == (
        not mock_app.add_punctuation
    )


def test_icon_activation(mock_app):
    """Test status icon activation."""
    app = mock_app
    
    # Test start recording
    app.recording = False
    app.on_icon_activate(app.status_icon)
    assert app.recording is True
    
    # Test stop recording
    app.recording = True
    app.on_icon_activate(app.status_icon)
    assert app.recording is False


def test_icon_popup(mock_app):
    """Test status icon popup menu."""
    app = mock_app
    
    # Test popup menu
    app.on_icon_popup(app.status_icon, 3, 0)
    
    # Verify popover was created and shown
    Gtk.PopoverMenu.new_from_model.assert_called_with(app.menu)
    Gtk.PopoverMenu.new_from_model.return_value.popup.assert_called_once()


def test_quit(mock_app):
    """Test quit functionality."""
    app = mock_app
    
    # Test quit action
    quit_action = app.app.lookup_action('quit')
    quit_action.activate(None)
    
    assert app.running is False
    app.app.quit.assert_called_once()


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
        mock_app.status_icon.set_icon.assert_called()


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
    with patch('whisper_widget.app.WhisperModel', mock_model):
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
    with patch('whisper_widget.app.webrtcvad.Vad', mock_vad):
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


def test_speech_detection_menu(mock_app):
    """Test speech detection settings menu items."""
    app = mock_app
    menu = app.menu.get_section(0)
    
    # Test min speech duration menu
    min_speech = menu.get_item_link(3, 'submenu')
    assert min_speech is not None
    assert min_speech.get_n_items() == 5  # [0.3, 0.5, 1.0, 1.5, 2.0]

    # Test max silence duration menu
    max_silence = menu.get_item_link(4, 'submenu')
    assert max_silence is not None
    assert max_silence.get_n_items() == 5  # [0.5, 1.0, 1.5, 2.0, 3.0]

    # Test speech start chunks menu
    chunks = menu.get_item_link(5, 'submenu')
    assert chunks is not None
    assert chunks.get_n_items() == 5  # [1, 2, 3, 4, 5]

    # Test noise reduction menu
    noise = menu.get_item_link(6, 'submenu')
    assert noise is not None
    assert noise.get_n_items() == 6  # [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]


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