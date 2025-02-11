#!/bin/bash

# Exit on error
set -e

echo "Setting up Whisper Widget development environment..."

# Install system dependencies
if [ -f /etc/debian_version ]; then
    echo "Installing Debian/Ubuntu dependencies..."
    sudo apt-get update
    sudo apt-get install -y \
        python3-venv \
        python3-dev \
        libgirepository1.0-dev \
        libcairo2-dev \
        pkg-config \
        python3-gi \
        python3-gi-cairo \
        gir1.2-gtk-4.0 \
        gir1.2-webkit2-4.1 \
        libayatana-appindicator3-dev \
        portaudio19-dev \
        python3-gst-1.0
elif [ -f /etc/fedora-release ]; then
    echo "Installing Fedora dependencies..."
    sudo dnf install -y \
        python3-devel \
        gobject-introspection-devel \
        cairo-devel \
        pkg-config \
        python3-gobject \
        gtk4-devel \
        webkit2gtk4.0-devel \
        libayatana-appindicator3-devel \
        portaudio-devel \
        gstreamer1-devel
fi

# Create and activate virtual environment
echo "Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

echo "Setup complete! Activate the virtual environment with:"
echo "source .venv/bin/activate" 