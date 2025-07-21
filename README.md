# Object Detection Suite with Ollama Vision & YOLO11

This collection contains two Python applications for object detection using local AI models:

1. **Live Webcam Detection** - Real-time object detection from webcam using Ollama + YOLO11
2. **Image Upload Detection** - Analyze uploaded images using local Gemma3:4b model

## Prerequisites

1. **Ollama installed and running**: Make sure you have Ollama installed and the vision model running:
   ```bash
   ollama serve
   ollama pull gemma3:4b
   ```

2. **Python 3.10+** with the required packages installed.

3. **YOLO11 Model** (for webcam app): Download YOLO11s.pt to `C:\Users\khoimn1\Downloads\yolo11s.pt`

## Installation

1. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

## Applications

### 1. Live Webcam Object Detection (`webcam_object_detection.py`)

Real-time object detection using webcam with dual detection systems.

**Run the application:**
```bash
python webcam_object_detection.py
```

**Features:**
- 🎥 **Live Camera Feed**: Real-time video display with privacy mode (smaller, transparent feed)
- 🔍 **Dual Detection Systems**: 
  - Ollama Gemma3:4b for detailed scene analysis
  - YOLO11 for fast, accurate object detection with bounding boxes
- ⚡ **Auto Detection**: YOLO11 continuous detection at 5 FPS with toggle
- 📊 **Visual Overlays**: Real-time bounding boxes with object labels
- 📈 **Performance Logging**: Detection timing analysis in milliseconds
- 🔄 **Manual Controls**: On-demand detection triggers

**Controls:**
- **Start/Stop Detection**: Toggle Ollama automatic object detection
- **Capture & Analyze**: Manual Ollama detection on current frame  
- **YOLO11 Detect**: Single YOLO detection with visual overlay
- **YOLO Auto**: Toggle continuous YOLO detection at 5 FPS
- **Privacy Mode**: Toggle camera feed transparency
- **Clear Overlay**: Remove detection bounding boxes
- **Detection Interval**: Adjust Ollama detection frequency

### 2. Image Upload Object Detection (`image_object_detection.py`)

Advanced object detection for uploaded images using local Gemma3:4b model.

**Run the application:**
```bash
python image_object_detection.py
```

**Features:**
- 📁 **Image Upload**: Support for JPG, PNG, BMP, GIF, TIFF, WebP formats
- 🖼️ **Image Preview**: Scrollable canvas with automatic scaling
- 🔍 **Multiple Analysis Modes**:
  - **Basic**: Simple object list
  - **Standard**: Specific descriptions  
  - **Detailed**: Colors, sizes, characteristics
  - **Comprehensive**: Complete analysis with materials and conditions
- 📍 **Position Detection**: Optional object location information
- 💾 **Results Management**: Save analysis to files, view history
- ⏱️ **Performance Tracking**: Processing time monitoring and logging

**Controls:**
- **📁 Upload Image**: Select image file for analysis
- **🔍 Analyze Objects**: Standard object detection
- **📋 Detailed Analysis**: Comprehensive scene analysis
- **💾 Save Results**: Export analysis to text file
- **🗑️ Clear Results**: Clear analysis history
- **Detail Level**: Choose analysis depth (Basic → Comprehensive)
- **Include Positions**: Toggle object location detection

## Quick Start Guide

### For Live Webcam Detection:
1. Start Ollama: `ollama serve`
2. Run: `python webcam_object_detection.py`
3. Click "YOLO Auto" for continuous detection
4. Use "Privacy Mode" for discretion

### For Image Analysis:
1. Start Ollama: `ollama serve`  
2. Run: `python image_object_detection.py`
3. Click "📁 Upload Image" and select your image
4. Choose detail level and click "🔍 Analyze Objects"
5. Save results with "💾 Save Results"

## System Requirements

- **Operating System**: Windows 10/11 (PowerShell)
- **Python**: 3.10 or higher
- **RAM**: 8GB+ recommended (4GB+ for YOLO11, 4GB+ for Gemma3:4b)
- **Storage**: 10GB+ free space for models
- **Camera**: USB/built-in webcam (for webcam app)

## Dependencies

```bash
# Core packages
pip install opencv-python
pip install pillow
pip install requests
pip install ultralytics  # For YOLO11
pip install numpy

# GUI packages  
pip install tkinter  # Usually included with Python
```

## File Structure

```
llmvision/
├── webcam_object_detection.py    # Live webcam detection app
├── image_object_detection.py     # Image upload detection app
├── logs/                         # Detection performance logs
│   ├── yolo_detection_times.log
│   └── gemma_detection_times.log
├── README.md
└── requirements.txt
```

## Troubleshooting

### General Issues
1. **Ollama connection failed**: 
   - Make sure Ollama is running: `ollama serve`
   - Check if running on localhost:11434
   - Verify Gemma3:4b model: `ollama list`

2. **Model not found**: 
   - Download Gemma3:4b: `ollama pull gemma3:4b`
   - For YOLO11: Download yolo11s.pt to `C:\Users\khoimn1\Downloads\`

### Webcam App Issues
3. **Camera not working**: 
   - Close other apps using the camera
   - Check camera permissions in Windows settings
   - Try different camera index if multiple cameras

4. **YOLO11 not loading**:
   - Verify yolo11s.pt file location
   - Check file permissions
   - Ensure sufficient RAM available

### Image App Issues  
5. **Image won't load**:
   - Check supported formats: JPG, PNG, BMP, GIF, TIFF, WebP
   - Verify file is not corrupted
   - Try smaller image size if memory issues

6. **Analysis too slow**:
   - Images are auto-resized to 1024px max
   - Use "Basic" detail level for faster results
   - Close other applications to free RAM

## Performance Notes

### Webcam Detection:
- **Camera Feed**: ~30 FPS display
- **YOLO11 Detection**: 100-200ms per frame
- **Ollama Detection**: 1-10 seconds per frame  
- **Auto YOLO**: Runs at 5 FPS (200ms intervals)

### Image Detection:
- **Gemma3:4b Analysis**: 2-15 seconds depending on detail level
- **Image Processing**: Auto-resize for optimal performance
- **Memory Usage**: ~2-4GB during analysis

## Logging

Both applications create detailed logs:
- **Location**: `logs/` directory
- **YOLO Logs**: `yolo_detection_times.log`
- **Gemma Logs**: `gemma_detection_times.log`
- **Content**: Timestamps, processing times, object counts

## Tips for Best Results

1. **Lighting**: Ensure good lighting for better detection accuracy
2. **Image Quality**: Higher resolution images give more detailed analysis  
3. **Detail Levels**: Start with "Standard" and increase as needed
4. **Performance**: Close unnecessary applications during analysis
5. **Privacy**: Use privacy mode for webcam discretion

## Model Information

- **Gemma3:4b**: Google's 4B parameter vision-language model
- **YOLO11s**: Ultralytics' latest object detection model (small version)
- **Both models run locally** - no internet required after setup
