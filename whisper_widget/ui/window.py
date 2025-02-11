"""GTK window module for the whisper widget application."""

# Set GTK version before ANY other imports
import gi
gi.require_version('Gtk', '3.0')
gi.require_version('WebKit2', '4.1')
gi.require_version('Gdk', '3.0')

import os
import pathlib
from typing import Optional, Callable

from gi.repository import Gtk, WebKit2, Gdk

from .menu import create_app_menu


class WhisperWindow(Gtk.ApplicationWindow):
    """Main application window."""

    def __init__(
        self,
        app: Gtk.Application,
        on_recording_start: Optional[Callable[[], None]] = None,
        on_recording_stop: Optional[Callable[[], None]] = None
    ):
        """Initialize the window."""
        super().__init__(application=app)
        
        # Set window properties
        self.set_title("Whisper Widget")
        self.set_default_size(400, 300)
        self.set_decorated(False)  # Make window borderless
        self.set_app_paintable(True)  # Enable transparency
        self.set_position(Gtk.WindowPosition.CENTER)
        
        # Store callbacks
        self.on_recording_start = on_recording_start
        self.on_recording_stop = on_recording_stop
        
        # Set up transparency
        self._setup_transparency()
        
        # Create layout
        self._create_layout()
        
        # Set up event handlers
        self._setup_events()

    def _setup_transparency(self) -> None:
        """Set up window transparency."""
        screen = self.get_screen()
        visual = screen.get_rgba_visual()
        if visual and screen.is_composited():
            self.set_visual(visual)
        
        css_provider = Gtk.CssProvider()
        css = b"""
            .transparent {
                background-color: rgba(0, 0, 0, 0);
                background-image: none;
                border-color: rgba(0, 0, 0, 0);
            }
        """
        css_provider.load_from_data(css)
        
        Gtk.StyleContext.add_provider_for_screen(
            screen,
            css_provider,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
        )
        
        # Make sure the window is truly transparent
        self.get_style_context().add_class('transparent')
        self.set_opacity(1.0)

    def _create_layout(self) -> None:
        """Create the window layout."""
        # Create main box with transparent background
        self.box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.box.set_size_request(400, 100)
        self.box.get_style_context().add_class('transparent')
        self.add(self.box)
        
        # Create WebView for visualization with transparent background
        self.webview = WebKit2.WebView()
        self.webview.set_background_color(Gdk.RGBA(0, 0, 0, 0))
        self.webview.set_size_request(400, 100)
        
        # Set WebView background to be transparent
        settings = WebKit2.Settings()
        settings.set_enable_webaudio(True)
        settings.set_enable_webgl(True)
        self.webview.set_settings(settings)
        
        # Load visualization HTML
        html_path = os.path.join(
            pathlib.Path(__file__).parent.parent,
            'static',
            'index.html'
        )
        self.webview.load_uri(f'file://{html_path}')
        
        # Add WebView to box
        self.box.pack_start(self.webview, True, True, 0)
        self.webview.get_style_context().add_class('transparent')

    def _setup_events(self) -> None:
        """Set up window event handlers."""
        # Right-click for menu
        self.connect('button-press-event', self._on_button_press)
        
        # Window dragging
        self.connect('button-press-event', self._on_drag_begin)
        self.connect('motion-notify-event', self._on_drag_update)
        self.connect('button-release-event', self._on_drag_end)

    def _on_button_press(self, widget: Gtk.Widget, event: Gdk.EventButton) -> bool:
        """Handle button press events."""
        if event.button == 3:  # Right mouse button
            # Create menu
            menu = create_app_menu(self.get_application())
            
            # Create menu widget from the menu model
            popup = Gtk.Menu.new_from_model(menu)
            popup.attach_to_widget(self, None)
            
            # Show all menu items
            popup.show_all()
            
            # Popup at mouse position
            popup.popup_at_pointer(event)
            return True
        
        return False

    def _on_drag_begin(self, widget: Gtk.Widget, event: Gdk.EventButton) -> bool:
        """Start window dragging."""
        if event.button == 1:  # Left mouse button
            self._drag_start_pos = (event.x_root, event.y_root)
            self._window_pos = self.get_position()
            return True
        return False

    def _on_drag_update(self, widget: Gtk.Widget, event: Gdk.EventMotion) -> bool:
        """Update window position during drag."""
        if hasattr(self, '_drag_start_pos') and hasattr(self, '_window_pos'):
            start_x, start_y = self._drag_start_pos
            win_x, win_y = self._window_pos
            
            dx = event.x_root - start_x
            dy = event.y_root - start_y
            
            self.move(int(win_x + dx), int(win_y + dy))
            return True
        return False

    def _on_drag_end(self, widget: Gtk.Widget, event: Gdk.EventButton) -> bool:
        """End window dragging."""
        if hasattr(self, '_drag_start_pos'):
            del self._drag_start_pos
            del self._window_pos
            return True
        return False

    def update_visualization(self, state: str) -> None:
        """Update the visualization state."""
        js = (
            'window.postMessage('
            f'{{"type": "state", "value": "{state}"}}, "*")'
        )
        self.webview.run_javascript(js, None, None, None) 