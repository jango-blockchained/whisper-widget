"""Menu creation and handling for the whisper widget application."""

# Set GTK version before ANY other imports
import gi
gi.require_version('Gtk', '3.0')

from gi.repository import Gtk, Gio, GLib


def create_app_menu(app: Gtk.Application) -> Gio.Menu:
    """Create the application menu."""
    menu = Gio.Menu.new()
    
    # Add transcription mode submenu
    trans_menu = _create_transcription_menu(app)
    menu.append_submenu('Transcription Mode', trans_menu)
    
    # Add output mode submenu
    output_menu = _create_output_menu(app)
    menu.append_submenu('Output Mode', output_menu)
    
    # Add model size submenu
    model_menu = _create_model_menu(app)
    menu.append_submenu('Model Size', model_menu)
    
    # Add language submenu
    lang_menu = _create_language_menu(app)
    menu.append_submenu('Language', lang_menu)
    
    # Add Speech Detection Settings submenu
    detect_menu = _create_detection_menu(app)
    menu.append_submenu('Speech Detection', detect_menu)
    
    # Add toggles
    menu.append('Auto-detect Speech', 'app.auto_detect')
    app.add_action(
        Gio.SimpleAction.new_stateful(
            'auto_detect',
            None,
            GLib.Variant.new_boolean(True)
        )
    )
    
    menu.append('Add Punctuation', 'app.add_punct')
    app.add_action(
        Gio.SimpleAction.new_stateful(
            'add_punct',
            None,
            GLib.Variant.new_boolean(True)
        )
    )
    
    menu.append('Wake Word Detection', 'app.wake_word_toggle')
    app.add_action(
        Gio.SimpleAction.new_stateful(
            'wake_word_toggle',
            None,
            GLib.Variant.new_boolean(False)
        )
    )
    
    # Add quit option
    menu.append('Quit', 'app.quit')
    quit_action = Gio.SimpleAction.new('quit', None)
    app.add_action(quit_action)
    
    return menu


def _create_transcription_menu(app: Gtk.Application) -> Gio.Menu:
    """Create transcription mode submenu."""
    menu = Gio.Menu.new()
    for mode in ['local', 'openai']:
        action = f'app.trans_mode_{mode}'
        action_obj = Gio.SimpleAction.new_stateful(
            f'trans_mode_{mode}',
            None,
            GLib.Variant.new_boolean(mode == 'local')
        )
        app.add_action(action_obj)
        menu.append(mode.capitalize(), action)
    return menu


def _create_output_menu(app: Gtk.Application) -> Gio.Menu:
    """Create output mode submenu."""
    menu = Gio.Menu.new()
    for mode in ['continuous', 'clipboard']:
        action = f'app.output_mode_{mode}'
        action_obj = Gio.SimpleAction.new_stateful(
            f'output_mode_{mode}',
            None,
            GLib.Variant.new_boolean(mode == 'continuous')
        )
        app.add_action(action_obj)
        menu.append(mode.capitalize(), action)
    return menu


def _create_model_menu(app: Gtk.Application) -> Gio.Menu:
    """Create model size submenu."""
    menu = Gio.Menu.new()
    for size in ['tiny', 'base', 'small', 'medium', 'large']:
        action = f'app.model_size_{size}'
        action_obj = Gio.SimpleAction.new_stateful(
            f'model_size_{size}',
            None,
            GLib.Variant.new_boolean(size == 'base')
        )
        app.add_action(action_obj)
        menu.append(size.capitalize(), action)
    return menu


def _create_language_menu(app: Gtk.Application) -> Gio.Menu:
    """Create language submenu."""
    menu = Gio.Menu.new()
    languages = [
        ('en', 'English'),
        ('fr', 'French'),
        ('de', 'German'),
        ('es', 'Spanish'),
        ('it', 'Italian'),
        ('pt', 'Portuguese'),
        ('nl', 'Dutch'),
        ('pl', 'Polish'),
        ('ru', 'Russian'),
        ('zh', 'Chinese'),
        ('ja', 'Japanese'),
        ('ko', 'Korean'),
    ]
    
    for code, name in languages:
        action = f'app.lang_{code}'
        action_obj = Gio.SimpleAction.new_stateful(
            f'lang_{code}',
            None,
            GLib.Variant.new_boolean(code == 'en')
        )
        app.add_action(action_obj)
        menu.append(name, action)
    return menu


def _create_detection_menu(app: Gtk.Application) -> Gio.Menu:
    """Create speech detection settings submenu."""
    menu = Gio.Menu.new()
    
    # VAD Sensitivity submenu
    vad_menu = Gio.Menu.new()
    for level in [1, 2, 3]:
        action = f'app.vad_level_{level}'
        action_obj = Gio.SimpleAction.new_stateful(
            f'vad_level_{level}',
            None,
            GLib.Variant.new_boolean(level == 2)
        )
        app.add_action(action_obj)
        vad_menu.append(f'Level {level}', action)
    menu.append_submenu('VAD Sensitivity', vad_menu)
    
    # Speech Start Settings
    start_menu = Gio.Menu.new()
    for chunks in [1, 2, 3, 4]:
        action = f'app.speech_start_{chunks}'
        action_obj = Gio.SimpleAction.new_stateful(
            f'speech_start_{chunks}',
            None,
            GLib.Variant.new_boolean(chunks == 2)
        )
        app.add_action(action_obj)
        start_menu.append(f'{chunks} chunk{"s" if chunks > 1 else ""}', action)
    menu.append_submenu('Speech Start Threshold', start_menu)
    
    # Silence Settings
    silence_menu = Gio.Menu.new()
    for threshold in [3, 5, 7, 10]:
        action = f'app.silence_threshold_{threshold}'
        action_obj = Gio.SimpleAction.new_stateful(
            f'silence_threshold_{threshold}',
            None,
            GLib.Variant.new_boolean(threshold == 7)
        )
        app.add_action(action_obj)
        silence_menu.append(f'{threshold} chunks', action)
    menu.append_submenu('Silence Threshold', silence_menu)
    
    # Duration Settings
    duration_menu = Gio.Menu.new()
    durations = [
        ('min_speech', 'Min Speech', [0.2, 0.5, 1.0], 0.5),
        ('max_silence', 'Max Silence', [0.5, 1.0, 2.0], 1.0),
        ('min_audio', 'Min Audio', [0.3, 0.5, 1.0], 0.5)
    ]
    
    for setting, label, values, default in durations:
        submenu = Gio.Menu.new()
        for value in values:
            action = f'app.{setting}_{str(value).replace(".", "_")}'
            action_obj = Gio.SimpleAction.new_stateful(
                f'{setting}_{str(value).replace(".", "_")}',
                None,
                GLib.Variant.new_boolean(abs(value - default) < 0.01)
            )
            app.add_action(action_obj)
            submenu.append(f'{value}s', action)
        duration_menu.append_submenu(label, submenu)
    menu.append_submenu('Duration Settings', duration_menu)
    
    return menu 