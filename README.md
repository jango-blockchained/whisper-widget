# Whisper Widget

[![Whisper Widget CI](https://github.com/jango-blockchained/whisper-widget/actions/workflows/ci.yml/badge.svg)](https://github.com/jango-blockchained/whisper-widget/actions/workflows/ci.yml)

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
- Ayatana AppIndicator3 (for system tray icon)

## Installation

1. Install system dependencies:
```bash
# For Ubuntu/Debian:
sudo apt-get install python3-dev portaudio19-dev python3-venv libgtk-4-dev \
    gobject-introspection libayatana-appindicator3-dev

# For Fedora:
sudo dnf install python3-devel portaudio-devel python3-virtualenv gtk4-devel \
    gobject-introspection-devel libayatana-appindicator3-devel
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

## Development and Testing

### Prerequisites

- Python 3.8+
- GTK 4.0
- PyGObject
- PyAudio

### Setup Development Environment

1. Clone the repository
```bash
git clone https://github.com/yourusername/whisper-widget.git
cd whisper-widget
```

2. Create a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
pip install -r tests/requirements-test.txt
```

### Running Tests

Run the full test suite with coverage and type checking:

```bash
# Run pytest with coverage
pytest tests/ --cov=whisper_widget --cov-report=term-missing

# Run mypy type checking
mypy whisper_widget

# Run linter
pylint whisper_widget
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Write tests for your changes
5. Ensure all tests pass
6. Submit a pull request

### Code Quality

- We use `pytest` for unit testing
- `mypy` for static type checking
- `pylint` for code linting
- Aim for at least 70% test coverage

### Continuous Integration

We use GitHub Actions for:
- Running tests on multiple Python versions
- Checking code coverage
- Type checking
- Linting

## License

MIT License 