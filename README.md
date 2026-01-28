# 🎯 Smart Surveillance System with YOLO

An enterprise-grade intelligent surveillance system featuring YOLOv8 object detection, ResNet-18 person re-identification, and anti-swap identity tracking with comprehensive logging.

## ✨ Features

- **🔍 Real-time Detection** - YOLOv8 nano for fast object detection (80 COCO classes)
- **👤 Person Re-ID** - ResNet-18 based 512-dimensional CNN embeddings for identity matching
- **🔄 Anti-Swap Logic** - Global best-match algorithm prevents identity swapping between people
- **📊 Entry/Exit Logging** - Automatic ENTERED/LEFT status tracking with timestamps
- **💾 Database Logging** - CSV database with 512 feature columns for each detection
- **🖥️ Enterprise UI** - Professional PyQt5 interface with dark theme
- **📹 Multi-Source** - Support for webcam, external camera, or video files
- **⚙️ Learning Mode** - Train the system to recognize specific individuals

## 🛠️ Tech Stack

- **GUI**: PyQt5
- **Detection**: YOLOv8 (Ultralytics)
- **Re-ID**: ResNet-18 (PyTorch)
- **Computer Vision**: OpenCV
- **Hardware Acceleration**: CUDA / MPS (Apple Silicon) / CPU

## 📦 Installation

```bash
# Install dependencies
pip install -r requirements.txt

# The YOLOv8 model (yolov8n.pt) is included in the repo
```

## 🚀 Usage

```bash
python main_and_database.py
```

### Controls
1. Select video source (webcam or file)
2. Click **Start Surveillance**
3. Use **MONITOR** tab to add people to track
4. Use **Learning Mode** to register new identities

## 📊 Database Output

The system logs to `surveillance_log.csv` with the following columns:
- `track_id` - Unique tracking ID
- `class_name` - Object class (person, car, etc.)
- `custom_label` - User-assigned name
- `confidence` - Detection confidence
- `timestamp` - Date and time
- `status` - ENTERED or LEFT
- `v0-v511` - 512-dimensional feature embedding

## ⚡ Performance

- **Detection FPS**: 20-30 FPS (depending on hardware)
- **Re-ID Threshold**: 0.80 cosine similarity
- **Missing Threshold**: 45 frames (~1.5s) before marking as LEFT

## 🔧 Configuration

Key parameters in `main_and_database.py`:
```python
detection_threshold = 0.5  # YOLO confidence
reid_threshold = 0.80      # Re-ID similarity
cooldown_time = 5          # Alert cooldown (seconds)
```

## 📄 License

MIT License
