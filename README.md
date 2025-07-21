# Live Object Detection with Ollama Vision

This Python application uses your local Ollama vision model to detect objects in real-time from your laptop's webcam.

## Prerequisites

1. **Ollama installed and running**: Make sure you have Ollama installed and the vision model running:
   ```bash
   ollama run gemma3:4b
   ```

2. **Python 3.10+** with the required packages installed.

## Installation

1. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Make sure Ollama is running with your vision model:
   ```bash
   ollama run gemma3:4b
   ```

2. Run the application:
   ```bash
   python webcam_object_detection.py
   ```

## Features

- **Live Camera Feed**: Real-time video display from your webcam
- **Auto Detection**: Automatically analyzes frames at configurable intervals
- **Manual Detection**: Capture and analyze current frame on demand
- **Detection History**: View all detection results with timestamps
- **Adjustable Interval**: Change how frequently auto-detection runs

## Controls

- **Start/Stop Detection**: Toggle automatic object detection
- **Capture & Analyze**: Manually trigger detection on current frame
- **Detection Interval**: Adjust time between automatic detections (in seconds)

## Troubleshooting

1. **Camera not working**: Make sure your webcam is not being used by another application
2. **Ollama connection failed**: Ensure Ollama is running on localhost:11434
3. **Model not found**: Verify that the gemma3:4b model is downloaded and available

## Notes

- The application resizes images to 512x384 pixels before sending to Ollama for faster processing
- Detection results are displayed in the right panel with timestamps
- The camera feed runs at ~30 FPS, while detection runs at the configured interval
