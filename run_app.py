#!/usr/bin/env python3

from whisper_widget.app import SpeechToTextApp


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