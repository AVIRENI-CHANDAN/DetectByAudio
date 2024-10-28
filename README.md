# Object Detection App with Audio Command Recognition

This project is a Python-based application that uses YOLO object detection integrated with an audio command feature. The application detects objects in real time via a webcam feed and allows the user to highlight detected objects by speaking a command.

## Overview

The Object Detection App leverages OpenCV and YOLO (You Only Look Once) for real-time object detection, while integrating audio input functionality with Google Speech Recognition API to process voice commands. When an object is detected that matches the spoken command, it is highlighted on the video feed.

## Features

- Real-time object detection using YOLO (YOLOv4 Tiny).
- Customizable class names for object detection.
- Voice command recognition to identify and highlight specific objects.
- GUI with a clickable button to start audio recording.
- Adjustable camera and audio recording settings.

## Project Structure

```plaintext
├── main.py                   # Entry point to run the application
├── application.py            # Core application logic and class-based implementation
├── util.py                   # Configuration constants and utility classes
├── classes.txt               # File containing class names for detected objects
├── Config/                   # Directory for YOLO model files
│   ├── yolov4-tiny.weights   # YOLO weights file
│   └── yolov4-tiny.cfg       # YOLO configuration file
└── README.md                 # Project documentation
```

## Requirements

- Python 3
- Libraries:
  - `opencv-python`
  - `numpy`
  - `sounddevice`
  - `soundfile`
  - `speechrecognition`
  - `scipy`
  - `pyaudio`

## Setup

1. **Install Dependencies**:
   Make sure to install the required dependencies via `pip`:

   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare YOLO Files**:
   Place the `yolov4-tiny.weights` and `yolov4-tiny.cfg` files in the `Config` directory, or modify paths in `util.py` if they are located elsewhere.

## Usage

1. **Run the App**:
   Start the application by running `main.py`:

   ```bash
   python main.py
   ```

2. **Interact with the Application**:

   - The webcam feed will display in a window named "Frame".
   - **Recording a Command**: Click inside the red "Record" button on the top left of the window to record audio.
   - **Highlighting Objects**: Speak the name of an object you want to highlight (e.g., "person", "car"). Detected objects matching the spoken command will be highlighted with a red rectangle.

3. **Exit**:
   - Press `q` to exit the application.

## Configuration

Configuration options are available in `util.py`:

- **`RecordBoxBoundaries`**: Adjusts the coordinates of the clickable recording button.
- **`YoloConfigFiles`**: Paths to YOLO model weight and configuration files.
- **`VideoCaptureWindowDimensions`**: Sets the dimensions of the video capture window.
- **`AudioRecordConfiguration`**: Configures audio sample rate and recording duration.

## Acknowledgments

- **YOLOv4 Tiny** for efficient real-time object detection.
- **Google Speech Recognition API** for speech-to-text functionality.
- **OpenCV** for handling video capture and rendering.
- **Numpy, Sounddevice, Soundfile** for audio processing and handling.