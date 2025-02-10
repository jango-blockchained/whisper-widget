#!/usr/bin/env python3

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
        transcription_mode="continuous",
        model_size="base",
        language="en",
        vad_sensitivity=3,
        auto_detect_speech=True,
        add_punctuation=True
    )
    
    # Run the application
    app.run()


if __name__ == "__main__":
    main() 