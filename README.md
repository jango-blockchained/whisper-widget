# Whisper Widget

A Linux system tray widget for real-time speech-to-text transcription using either local Whisper models (via faster-whisper) or OpenAI's API.

## Features

- Modern GTK 4 user interface
- System tray icon for easy control
- Voice Activity Detection (VAD) for automatic speech detection
- Optional noise reduction
- Support for both local and OpenAI transcription backends
- Hotkey support (F9) for manual recording control
- Continuous background operation

## Requirements

- Python 3.8+
- Linux system with audio input
- GTK 4.0+ and GObject Introspection
- PortAudio development files (for PyAudio)

## Installation

1. Install system dependencies:
```bash
# For Ubuntu/Debian:
sudo apt-get install python3-dev portaudio19-dev python3-venv libgtk-4-dev gobject-introspection

# For Fedora:
sudo dnf install python3-devel portaudio-devel python3-virtualenv gtk4-devel gobject-introspection-devel
```

2. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the application:
```bash
python run_app.py
```

2. The application will appear in your system tray
3. Use the tray icon menu or press F9 to start/stop recording
4. Transcriptions will be printed to the console

## Configuration

- Default configuration uses local transcription with the "base" Whisper model
- To use OpenAI's API, modify the initialization in `whisper_widget.py`:
  ```python
  app = SpeechToTextApp(transcription_mode="openai", openai_api_key="your-api-key")
  ```

## Testing

1. Install test dependencies:
```bash
pip install -r tests/requirements-test.txt
```

2. Run tests:
```bash
# Run all tests
pytest

# Run tests with coverage report
pytest --cov=whisper_widget

# Run specific test file
pytest tests/test_settings.py

# Run tests in verbose mode
pytest -v
```

### Test Structure

- `tests/conftest.py`: Common test fixtures and configuration
- `tests/test_settings.py`: Tests for settings management
- `tests/test_audio.py`: Tests for audio recording and processing
- `tests/test_ui.py`: Tests for UI components and interactions

### Writing Tests

When adding new features, please ensure:
1. Unit tests cover the new functionality
2. Integration tests verify feature interactions
3. Mock external dependencies (audio, UI, etc.)
4. Use fixtures from conftest.py when possible

## License

MIT License 