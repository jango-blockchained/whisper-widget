"""GTK window module for the whisper widget application."""

import os
import pathlib
from typing import Optional, Callable
import gi

gi.require_version('Gtk', '3.0')
gi.require_version('WebKit2', '4.0')
gi.require_version('Gdk', '3.0')
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
        
        self.set_title("Whisper Widget")
        self.set_default_size(400, 300)
        self.set_decorated(False)  # Make window borderless
        
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
        css_provider = Gtk.CssProvider()
        css_provider.load_from_data(
            b"""
            window {
                background-color: transparent;
            }
            box {
                min-height: 100px;
            }
            """
        )
        
        Gtk.StyleContext.add_provider_for_display(
            Gdk.Display.get_default(),
            css_provider,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
        )

    def _create_layout(self) -> None:
        """Create the window layout."""
        # Create main box
        self.box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.box.set_size_request(400, 100)
        self.set_child(self.box)
        
        # Create WebView for visualization
        self.webview = WebKit2.WebView.new()
        self.webview.set_background_color(Gdk.RGBA())
        self.webview.set_size_request(400, 100)
        
        # Load visualization HTML
        html_path = os.path.join(
            pathlib.Path(__file__).parent.parent,
            'static',
            'index.html'
        )
        self.webview.load_uri(f'file://{html_path}')
        
        # Add WebView to box
        self.box.append(self.webview)

    def _setup_events(self) -> None:
        """Set up window event handlers."""
        # Right-click for menu
        right_click = Gtk.GestureClick.new()
        right_click.set_button(3)  # Right mouse button
        right_click.connect('pressed', self._on_right_click)
        self.add_controller(right_click)
        
        # Window dragging
        drag = Gtk.GestureDrag.new()
        drag.connect('drag-begin', self._on_drag_begin)
        drag.connect('drag-update', self._on_drag_update)
        self.add_controller(drag)

    def _on_right_click(
        self,
        gesture: Gtk.GestureClick,
        n_press: int,
        x: float,
        y: float
    ) -> None:
        """Handle right-click to show menu."""
        # Create menu
        menu = create_app_menu(self.get_application())
        
        # Create popover
        popover = Gtk.PopoverMenu.new_from_model(menu)
        popover.set_pointing_to(Gdk.Rectangle(x=x, y=y, width=1, height=1))
        popover.popup()

    def _on_drag_begin(
        self,
        gesture: Gtk.GestureDrag,
        start_x: float,
        start_y: float
    ) -> None:
        """Start window dragging."""
        self._drag_start_pos = self.get_position()

    def _on_drag_update(
        self,
        gesture: Gtk.GestureDrag,
        offset_x: float,
        offset_y: float
    ) -> None:
        """Update window position during drag."""
        if hasattr(self, '_drag_start_pos'):
            start_x, start_y = self._drag_start_pos
            self.move(
                int(start_x + offset_x),
                int(start_y + offset_y)
            )

    def update_visualization(self, state: str) -> None:
        """Update the visualization state."""
        js = (
            'window.postMessage('
            f'{{"type": "state", "value": "{state}"}}, "*")'
        )
        self.webview.evaluate_javascript(js, -1, None, None, None) 