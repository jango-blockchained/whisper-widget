#!/usr/bin/env python3

# Set GTK version before ANY other imports
import gi
gi.require_version('Gtk', '3.0')
gi.require_version('Gdk', '3.0')
gi.require_version('WebKit2', '4.1')

import os
import sys
import warnings
from whisper_widget.app import SpeechToTextApp

# Suppress ALSA warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
os.environ['ALSA_CARD'] = 'Generic'

# Redirect stderr to devnull temporarily to suppress ALSA warnings
stderr = sys.stderr
with open(os.devnull, 'w') as devnull:
    sys.stderr = devnull
    try:
        import sounddevice as sd
        import pyaudio
    finally:
        sys.stderr = stderr


def main():
    # Initialize the app with default settings
    app = SpeechToTextApp(
        transcription_mode="local",
        output_mode="clipboard",
        model_size="base",
        language="en",
        vad_sensitivity=3,
        auto_detect_speech=True,
        add_punctuation=True,
        use_tray=True  # Disable system tray by default
    )
    
    # Run the application
    app.run()


if __name__ == "__main__":
    main() 