"""
INTEGRATED SURVEILLANCE SYSTEM (V3 - Anti-Swap Logic)
Combines Enterprise Surveillance UI with Global Best-Match Re-ID.

FIXES:
- Identity Swapping: Now uses a global scoring matrix to assign IDs. 
  It calculates all possible matches first and picks the highest scores 
  globally, preventing one person from "stealing" another's ID.

PRESERVES:
- All Database Logging (512 columns)
- Entry/Exit Logic
- UI Features
"""

import sys
import cv2
import numpy as np
import csv
import os
import json
import threading
import random
from datetime import datetime

# GUI Imports
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QComboBox, 
                             QGroupBox, QListWidget, QTextEdit, QMessageBox,
                             QFileDialog, QSlider, QFrame, QScrollArea, 
                             QListWidgetItem, QLineEdit, QCheckBox, QSpinBox,
                             QTabWidget, QProgressBar, QInputDialog, QStyleFactory,
                             QDialog, QDialogButtonBox)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer, QPropertyAnimation, QEasingCurve, QSize
from PyQt5.QtGui import QImage, QPixmap, QFont, QPalette, QColor, QIcon, QPainter, QPen

# Deep Learning Imports
import torch
import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights
from ultralytics import YOLO

# --- Configuration & Theme Constants ---
THEME = {
    "bg_dark": "#0b0c10",       # Deepest Black/Blue
    "bg_panel": "#1f2833",      # Dark Slate
    "accent_primary": "#66fcf1", # Cyan/Electric Blue
    "accent_secondary": "#45a29e", # Muted Cyan
    "text_light": "#ffffff",
    "text_dim": "#c5c6c7",
    "danger": "#ff4d4d",        # Alert Red
    "success": "#00e676",       # Bright Green
    "border": "#2d3540"
}

# ==========================================
# PART 1: DATABASE MANAGER (512 Columns)
# ==========================================
class DatabaseManager:
    def __init__(self, filename="surveillance_log.csv"):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.filename = os.path.join(base_dir, filename)
        print(f"--- DATABASE LOCATION: {self.filename} ---")
        self.init_db()

    def init_db(self):
        """Initialize the CSV file with headers ONLY if it doesn't exist"""
        if not os.path.exists(self.filename):
            try:
                with open(self.filename, mode='w', newline='', encoding='utf-8') as file:
                    writer = csv.writer(file)
                    
                    # Base Headers
                    headers = ['track_id', 'class_name', 'custom_label', 'confidence', 'timestamp', 'status']
                    
                    # Add 512 columns for the embedding vector features
                    feature_headers = [f"v{i}" for i in range(512)]
                    
                    writer.writerow(headers + feature_headers)
                print(f"CSV Log created successfully with 512 feature columns.")
            except Exception as e:
                print(f"CSV Init Error: {e}")
        else:
            print(f"CSV Log found. New data will be appended.")

    def log_event(self, track_id, class_name, custom_label, confidence, status, embedding):
        """
        Logs a detection event (ENTERED/LEFT) to the CSV file.
        Spreads the embedding vector across columns.
        """
        try:
            label_to_store = custom_label if custom_label else "Unknown"
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Prepare the embedding list (512 floats) or empty placeholders
            if embedding is not None:
                # Ensure it's a flat list
                if hasattr(embedding, 'flatten'):
                    embed_list = list(np.round(embedding.flatten(), 5))
                else:
                    embed_list = list(np.round(embedding, 5))
            else:
                embed_list = [""] * 512

            # Base Data
            row_data = [
                track_id, 
                class_name, 
                label_to_store, 
                f"{confidence:.2f}" if confidence else "0.00", 
                timestamp, 
                status
            ]
            
            # Combine Base + Features
            full_row = row_data + embed_list

            with open(self.filename, mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(full_row)
            
            print(f"DB LOG: {label_to_store} -> {status}")
                
        except Exception as e:
            print(f"CSV Logging Error: {e}")

# ==========================================
# PART 2: CORE AI & VIDEO LOGIC
# ==========================================

class DeepFeatureExtractor:
    """
    CNN-based Feature Extractor using ResNet-18.
    Generates a 512-dimensional embedding vector for visual identity.
    """
    def __init__(self):
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print("Using NVIDIA CUDA GPU")
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps') 
            print("Using Apple Metal (MPS) Acceleration")
        else:
            self.device = torch.device('cpu')
            print("Using CPU (Warning: Slower)")
            
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        # Remove the final classification layer (fc) to get the feature vector
        self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))
        self.model.to(self.device)
        self.model.eval() 
        
        self.transforms = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def extract(self, img_numpy):
        try:
            img_rgb = cv2.cvtColor(img_numpy, cv2.COLOR_BGR2RGB)
            img_tensor = self.transforms(img_rgb).unsqueeze(0).to(self.device)
            with torch.no_grad():
                features = self.model(img_tensor)
                features = features.view(-1)
            return features.cpu().numpy()
        except Exception as e:
            print(f"Extraction Error: {e}")
            return None

class VideoThread(QThread):
    """
    Background Thread for Video Processing & AI Detection.
    """
    change_pixmap_signal = pyqtSignal(np.ndarray)
    alert_signal = pyqtSignal(str, str)
    detection_signal = pyqtSignal(dict)
    fps_signal = pyqtSignal(int)
    learn_signal = pyqtSignal(dict) 
    
    def __init__(self):
        super().__init__()
        # Initialize Database
        self.db = DatabaseManager()
        
        self.running = False
        self.cap = None
        self.model = None
        self.reid_model = None 
        
        # Structure: { 'Custom Name': { 'id': 12, 'class': 'person', 'status': 'ACTIVE', ... } }
        self.monitored_objects = {}
        
        self.detection_threshold = 0.5
        self.reid_threshold = 0.80 
        self.alert_cooldown = {}
        self.cooldown_time = 5
        self.fps = 0
        self.video_file = None
        self.show_labels = True
        self.show_confidence = True
        self.color_cache = {}
        
        # Learning Mode Flags
        self.learning_mode = False
        self.learning_frames = 0
        
    def set_video_source(self, source):
        if self.cap is not None:
            self.cap.release()
        
        if isinstance(source, str):
            self.cap = cv2.VideoCapture(source)
            self.video_file = source
        else:
            self.cap = cv2.VideoCapture(source)
            
        return self.cap.isOpened()
    
    def load_model(self):
        try:
            self.model = YOLO('yolov8n.pt')
            if self.reid_model is None:
                self.reid_model = DeepFeatureExtractor()
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def start_learning_mode(self):
        self.learning_mode = True
        self.learning_frames = 0
    
    def add_monitored_object(self, user_name, class_name, track_id, embedding=None):
        # When adding manually, we set status to MISSING initially so the logic detects the "Entry"
        self.monitored_objects[user_name] = {
            'id': track_id, 
            'class': class_name,
            'status': 'MISSING', # Start as missing to trigger "ENTERED" log immediately
            'last_seen': datetime.now(),
            'missing_frames': 0,
            'embedding': embedding,
            'latest_frame_embedding': None,
            'latest_conf': 0.0
        }
    
    def remove_monitored_object_by_name(self, name):
        if name in self.monitored_objects:
            del self.monitored_objects[name]
    
    def get_object_color(self, id_or_class):
        key = str(id_or_class)
        if key in self.color_cache:
            return self.color_cache[key]
        random.seed(key)
        r = random.randint(50, 255)
        g = random.randint(50, 255)
        b = random.randint(50, 255)
        if abs(r-g) < 20 and abs(g-b) < 20:
            b = (b + 100) % 255
        color = (b, g, r) 
        self.color_cache[key] = color
        return color

    def get_deep_embedding(self, image, box):
        x1, y1, x2, y2 = map(int, box)
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        roi = image[y1:y2, x1:x2]
        if roi.size == 0: return None
        return self.reid_model.extract(roi)

    def compute_similarity(self, embed1, embed2):
        if embed1 is None or embed2 is None: return 0.0
        norm1 = np.linalg.norm(embed1)
        norm2 = np.linalg.norm(embed2)
        if norm1 == 0 or norm2 == 0: return 0.0
        return np.dot(embed1, embed2) / (norm1 * norm2)

    def run(self):
        """Main video processing loop"""
        self.running = True
        frame_count = 0
        start_time = datetime.now()
        
        if self.model is None:
            if not self.load_model():
                return
        
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                break
                
            ret, frame = self.cap.read()
            if not ret:
                if self.video_file:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    break
            
            frame_count += 1
            if frame_count % 30 == 0:
                elapsed = (datetime.now() - start_time).total_seconds()
                self.fps = int(30 / elapsed) if elapsed > 0 else 0
                self.fps_signal.emit(self.fps)
                start_time = datetime.now()
            
            # Run YOLO Tracking
            results = self.model.track(frame, conf=self.detection_threshold, persist=True, verbose=False)
            
            # Process detections
            current_ids = []
            detected_classes = {}
            annotated_frame = frame.copy()
            overlay = annotated_frame.copy()
            current_frame_detections = []
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    class_name = self.model.names[cls]
                    track_id = int(box.id[0]) if box.id is not None else None
                    
                    if track_id is not None:
                        current_ids.append(track_id)
                    
                    # Store for potential learning & re-id
                    detection_info = {
                        'class': class_name,
                        'id': track_id,
                        'conf': conf,
                        'area': (x2-x1) * (y2-y1),
                        'box': (x1, y1, x2, y2)
                    }
                    current_frame_detections.append(detection_info)
                    
                    detected_classes[class_name] = detected_classes.get(class_name, 0) + 1
            
            # --- PHASE 1: DIRECT ID MATCHING (Fastest) ---
            # Identify which monitored objects are ALREADY tracked by ID
            
            active_track_ids = [] # IDs that are definitely claimed by monitored objects
            
            for user_name, obj_data in self.monitored_objects.items():
                target_id = obj_data['id']
                
                # Reset ephemeral data
                obj_data['is_present_this_frame'] = False
                
                if target_id is not None and target_id in current_ids:
                    # Found via simple ID persistence
                    obj_data['is_present_this_frame'] = True
                    active_track_ids.append(target_id)
                    
                    # Update embedding for freshness (helps if appearance changes slightly)
                    # Find the specific detection
                    for det in current_frame_detections:
                        if det['id'] == target_id:
                            # Optimize: only extract occasionally to save FPS, or every frame for max accuracy?
                            # Let's extract every frame for stability during movement
                            emb = self.get_deep_embedding(frame, det['box'])
                            obj_data['latest_frame_embedding'] = emb
                            obj_data['latest_conf'] = det['conf']
                            break

            # --- PHASE 2: GLOBAL BEST-MATCH RE-IDENTIFICATION ---
            # For objects that are MISSING (not found by ID), try to match visually.
            # We solve this globally: Find ALL missing people vs ALL unclaimed detections.
            
            missing_targets = [] # List of (user_name, obj_data)
            for user_name, obj_data in self.monitored_objects.items():
                if not obj_data['is_present_this_frame'] and obj_data.get('embedding') is not None:
                    if obj_data['status'] == "MISSING": # Only run expensive Re-ID if actively missing
                         missing_targets.append((user_name, obj_data))

            # Filter detections: Only consider IDs that aren't already claimed by Phase 1
            unclaimed_detections = [det for det in current_frame_detections if det['id'] not in active_track_ids]
            
            # Proceed only if we have both missing people and unclaimed detections
            if missing_targets and unclaimed_detections:
                
                # Calculate Similarity Matrix
                # Structure: [ {'score': 0.9, 'target_idx': 0, 'det_idx': 1}, ... ]
                matches = []
                
                for t_idx, (t_name, t_data) in enumerate(missing_targets):
                    for d_idx, det in enumerate(unclaimed_detections):
                        
                        # Optimization: Only match same class (Person vs Person)
                        if det['class'] != t_data['class']:
                            continue
                            
                        # Extract feature for candidate (Expensive!)
                        # We do this here. If FPS drops, we can optimize to only do it once per unique ID per sec.
                        cand_embed = self.get_deep_embedding(frame, det['box'])
                        
                        # Cache it in the detection dict so we don't re-compute if loop logic changes
                        det['embedding'] = cand_embed 
                        
                        score = self.compute_similarity(t_data['embedding'], cand_embed)
                        
                        if score > self.reid_threshold:
                            matches.append({
                                'score': score,
                                'target': t_data,
                                'target_name': t_name,
                                'detection': det
                            })
                
                # GLOBAL SORT: Sort all potential matches by score (Highest first)
                matches.sort(key=lambda x: x['score'], reverse=True)
                
                matched_targets = set()
                matched_detections = set()
                
                # Assign matches greedily from the sorted list
                for match in matches:
                    t_name = match['target_name']
                    d_id = match['detection']['id']
                    
                    # If this target OR this detection has already been assigned in a better match, skip
                    if t_name in matched_targets or d_id in matched_detections:
                        continue
                        
                    # --- EXECUTE MATCH ---
                    t_data = match['target']
                    old_id = t_data['id']
                    
                    t_data['id'] = d_id
                    t_data['is_present_this_frame'] = True
                    t_data['latest_frame_embedding'] = match['detection']['embedding']
                    t_data['latest_conf'] = match['detection']['conf']
                    
                    # Mark as used
                    matched_targets.add(t_name)
                    matched_detections.add(d_id)
                    
                    # Alert
                    if self.check_cooldown(f"{t_name}_reid", datetime.now()):
                         self.alert_signal.emit(f"AI RE-ID: FOUND '{t_name}' (ID {old_id} -> {d_id}, Score: {match['score']:.2f})", datetime.now().strftime("%H:%M:%S"))

            # --- PHASE 3: DRAWING & LOGGING ---
            
            for result in results: # Re-loop for drawing to ensure overlays are correct
                boxes = result.boxes
                for box in boxes:
                    track_id = int(box.id[0]) if box.id is not None else None
                    if track_id is None: continue
                    
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0])
                    class_name = self.model.names[cls]
                    
                    # Determine Identity
                    display_name = ""
                    is_target = False
                    
                    for user_name, obj_data in self.monitored_objects.items():
                        if obj_data['id'] == track_id:
                            is_target = True
                            display_name = user_name
                            break
                        # Also handle general class tracking (non-ID specific)
                        elif obj_data['id'] is None and obj_data['class'] == class_name:
                            is_target = True
                            display_name = user_name
                            break
                            
                    if is_target:
                        color = (255, 255, 0)
                        thickness = 3
                        full_label = f"{display_name.upper()} [{track_id}]"
                    else:
                        color = self.get_object_color(track_id)
                        thickness = 2
                        full_label = f"{class_name.upper()} {track_id}"

                    # Draw
                    self.draw_tech_corners(overlay, x1, y1, x2, y2, color, thickness)
                    if self.show_labels:
                        (label_w, label_h), _ = cv2.getTextSize(full_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        cv2.rectangle(overlay, (x1, y1 - 25), (x1 + label_w + 10, y1), (0, 0, 0), -1)
                        cv2.line(overlay, (x1, y1-25), (x1 + label_w + 10, y1-25), color, 1)
                        cv2.putText(overlay, full_label, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

            # --- PHASE 4: STATE MANAGEMENT & DATABASE ---
            
            current_time = datetime.now()
            
            for user_name, obj_data in self.monitored_objects.items():
                previous_status = obj_data['status']
                
                # Check general class presence if ID didn't match
                if obj_data['id'] is None: 
                    # If tracking "All cars", check if any car was seen
                     if detected_classes.get(obj_data['class'], 0) > 0:
                         obj_data['is_present_this_frame'] = True
                
                if obj_data['is_present_this_frame']:
                    obj_data['status'] = "ACTIVE"
                    obj_data['missing_frames'] = 0
                    obj_data['last_seen'] = current_time
                    
                    # LOG ENTERED
                    if previous_status == "MISSING":
                        # Use the fresh embedding we captured this frame
                        emb_to_log = obj_data.get('latest_frame_embedding')
                        # Fallback to reference if fresh one failed for some reason
                        if emb_to_log is None: emb_to_log = obj_data.get('embedding')
                        
                        self.db.log_event(
                            track_id=obj_data['id'],
                            class_name=obj_data['class'],
                            custom_label=user_name,
                            confidence=obj_data.get('latest_conf', 0.0),
                            status="ENTERED",
                            embedding=emb_to_log
                        )
                else:
                    obj_data['missing_frames'] += 1
                    if obj_data['missing_frames'] > 45: # 1.5s buffer
                        obj_data['status'] = "MISSING"
                        
                        # LOG LEFT
                        if previous_status == "ACTIVE":
                            if self.check_cooldown(user_name, current_time):
                                self.trigger_alert(f"{user_name}", f"MISSING (ID: {obj_data['id']})")
                                
                            # Use reference embedding because object is gone
                            self.db.log_event(
                                track_id=obj_data['id'],
                                class_name=obj_data['class'],
                                custom_label=user_name,
                                confidence=0.0,
                                status="LEFT",
                                embedding=obj_data.get('embedding')
                            )


            # Learning Mode Logic (Unchanged)
            if self.learning_mode:
                self.learning_frames += 1
                if current_frame_detections:
                    candidates = []
                    for det in current_frame_detections:
                        embedding = self.get_deep_embedding(frame, det['box'])
                        det['embedding'] = embedding
                        candidates.append(det)
                    self.learn_signal.emit({'status': 'candidates', 'candidates': candidates})
                    self.learning_mode = False 
                elif self.learning_frames > 60:
                    self.learn_signal.emit({'status': 'timeout'})
                    self.learning_mode = False

            annotated_frame = cv2.addWeighted(overlay, 1.0, annotated_frame, 0.0, 0)
            
            # Emit Stats
            detection_stats = {
                'total_objects': len(current_frame_detections),
                'unique_objects': len(detected_classes),
                'monitored_data': {
                    name: {
                        'status': data['status'],
                        'info': f"{data['class']} #{data['id']}" if data['id'] else data['class']
                    } for name, data in self.monitored_objects.items()
                }
            }
            self.detection_signal.emit(detection_stats)
            self.change_pixmap_signal.emit(annotated_frame)
        
        if self.cap is not None:
            self.cap.release()
            
    def draw_tech_corners(self, img, x1, y1, x2, y2, color, thickness=1):
        len_line = int(min(x2-x1, y2-y1) * 0.2)
        cv2.line(img, (x1, y1), (x1 + len_line, y1), color, thickness)
        cv2.line(img, (x1, y1), (x1, y1 + len_line), color, thickness)
        cv2.line(img, (x2, y1), (x2 - len_line, y1), color, thickness)
        cv2.line(img, (x2, y1), (x2, y1 + len_line), color, thickness)
        cv2.line(img, (x1, y2), (x1 + len_line, y2), color, thickness)
        cv2.line(img, (x1, y2), (x1, y2 - len_line), color, thickness)
        cv2.line(img, (x2, y2), (x2 - len_line, y2), color, thickness)
        cv2.line(img, (x2, y2), (x2, y2 - len_line), color, thickness)

    def check_cooldown(self, key, current_time):
        if key not in self.alert_cooldown:
            self.alert_cooldown[key] = current_time
            return True
        if (current_time - self.alert_cooldown[key]).total_seconds() > self.cooldown_time:
            self.alert_cooldown[key] = current_time
            return True
        return False

    def trigger_alert(self, object_name, alert_type):
        timestamp = datetime.now().strftime("%H:%M:%S")
        message = f"{object_name.upper()}: {alert_type}"
        self.alert_signal.emit(message, timestamp)
        threading.Thread(target=self.play_alert_sound, daemon=True).start()
    
    def play_alert_sound(self):
        try:
            print('\a') 
        except:
            pass
    
    def stop(self):
        self.running = False
        self.wait()


class SurveillanceSystem(QMainWindow):
    """
    Main Application Window - Enterprise Grade UI
    """
    
    def __init__(self):
        super().__init__()
        self.video_thread = VideoThread()
        self.alert_log = []
        
        self.all_objects = [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
            "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
            "hair drier", "toothbrush"
        ]
        
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle("Surveillance Command Center | Enterprise V3 (Anti-Swap)")
        self.setGeometry(50, 50, 1600, 950)
        self.apply_theme()
        
        central_widget = QWidget()
        central_widget.setStyleSheet(f"background-color: {THEME['bg_dark']};")
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        header = self.create_professional_header()
        main_layout.addWidget(header)
        
        body_layout = QHBoxLayout()
        body_layout.setContentsMargins(15, 15, 15, 15)
        body_layout.setSpacing(15)
        
        left_panel = self.create_control_panel()
        body_layout.addWidget(left_panel)
        
        video_panel = self.create_video_display()
        body_layout.addWidget(video_panel, stretch=1)
        
        right_panel = self.create_info_panel()
        body_layout.addWidget(right_panel)
        
        main_layout.addLayout(body_layout)
        
        status_bar = self.create_status_bar()
        main_layout.addWidget(status_bar)
        
        self.video_thread.change_pixmap_signal.connect(self.update_frame)
        self.video_thread.alert_signal.connect(self.handle_alert)
        self.video_thread.detection_signal.connect(self.update_statistics)
        self.video_thread.fps_signal.connect(self.update_fps)
        self.video_thread.learn_signal.connect(self.handle_learn_result)
        
    def apply_theme(self):
        app = QApplication.instance()
        app.setStyle("Fusion")
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(THEME["bg_dark"]))
        palette.setColor(QPalette.WindowText, QColor(THEME["text_light"]))
        palette.setColor(QPalette.Base, QColor(THEME["bg_panel"]))
        palette.setColor(QPalette.AlternateBase, QColor(THEME["bg_dark"]))
        palette.setColor(QPalette.Text, QColor(THEME["text_light"]))
        palette.setColor(QPalette.Button, QColor(THEME["bg_panel"]))
        palette.setColor(QPalette.ButtonText, QColor(THEME["text_light"]))
        palette.setColor(QPalette.Highlight, QColor(THEME["accent_primary"]))
        palette.setColor(QPalette.HighlightedText, QColor("#000000"))
        app.setPalette(palette)
        
    def create_professional_header(self):
        header = QFrame()
        header.setFixedHeight(70)
        header.setStyleSheet(f"""
            QFrame {{
                background-color: {THEME['bg_panel']};
                border-bottom: 2px solid {THEME['accent_primary']};
            }}
        """)
        
        layout = QHBoxLayout(header)
        layout.setContentsMargins(25, 10, 25, 10)
        
        title_layout = QVBoxLayout()
        main_title = QLabel("SURVEILLANCE COMMAND CENTER")
        main_title.setStyleSheet(f"""
            font-family: 'Arial', sans-serif;
            font-size: 24px;
            font-weight: 800;
            color: {THEME['text_light']};
            letter-spacing: 2px;
        """)
        
        subtitle = QLabel("ResNet Powered Identity Detection")
        subtitle.setStyleSheet(f"""
            font-size: 11px;
            font-weight: 600;
            color: {THEME['accent_secondary']};
            letter-spacing: 3px;
        """)
        
        title_layout.addWidget(main_title)
        title_layout.addWidget(subtitle)
        layout.addLayout(title_layout)
        layout.addStretch()
        
        self.system_status = QLabel("SYSTEM STANDBY")
        self.system_status.setStyleSheet(f"""
            font-size: 12px;
            font-weight: bold;
            color: {THEME['text_dim']};
            background-color: rgba(255, 255, 255, 0.05);
            padding: 8px 16px;
            border-radius: 4px;
            border: 1px solid {THEME['text_dim']};
        """)
        layout.addWidget(self.system_status)
        return header
    
    def create_control_panel(self):
        panel = QFrame()
        panel.setFixedWidth(380)
        panel.setStyleSheet(f"""
            QFrame {{
                background-color: {THEME['bg_panel']};
                border: 1px solid {THEME['border']};
                border-radius: 4px;
            }}
        """)
        
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        
        tabs = QTabWidget()
        tabs.setStyleSheet(f"""
            QTabWidget::pane {{ border: none; background-color: transparent; }}
            QTabBar::tab {{
                background-color: {THEME['bg_dark']};
                color: {THEME['text_dim']};
                padding: 12px 25px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                font-weight: bold;
                margin-right: 1px;
            }}
            QTabBar::tab:selected {{
                background-color: {THEME['accent_primary']};
                color: {THEME['bg_dark']};
            }}
        """)
        
        tabs.addTab(self.create_source_tab(), "SOURCE")
        tabs.addTab(self.create_monitor_tab(), "MONITOR")
        tabs.addTab(self.create_settings_tab(), "SETTINGS")
        
        layout.addWidget(tabs)
        return panel
    
    def create_source_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(15, 20, 15, 20)
        layout.setSpacing(20)
        
        source_group = self.create_card_group("VIDEO INPUT")
        s_layout = QVBoxLayout()
        
        self.source_combo = QComboBox()
        self.source_combo.addItems([
            "Camera 0 (Integrated)",
            "Camera 1 (External)",
            "Video File (Select below)"
        ])
        self.source_combo.setStyleSheet(self.get_combo_style())
        s_layout.addWidget(self.source_combo)
        
        file_layout = QHBoxLayout()
        self.btn_browse = self.create_premium_button("BROWSE FILES")
        self.btn_browse.clicked.connect(self.browse_video)
        file_layout.addWidget(self.btn_browse)
        s_layout.addLayout(file_layout)
        
        self.file_label = QLabel("No file loaded")
        self.file_label.setStyleSheet(f"color: {THEME['text_dim']}; font-style: italic; font-size: 11px;")
        s_layout.addWidget(self.file_label)
        
        source_group.setLayout(s_layout)
        layout.addWidget(source_group)
        
        control_group = self.create_card_group("SYSTEM ACTIVATION")
        c_layout = QVBoxLayout()
        
        self.btn_start = self.create_premium_button("INITIATE SURVEILLANCE", primary=True)
        self.btn_start.clicked.connect(self.start_monitoring)
        c_layout.addWidget(self.btn_start)
        
        self.btn_stop = self.create_premium_button("TERMINATE SURVEILLANCE", danger=True)
        self.btn_stop.clicked.connect(self.stop_monitoring)
        self.btn_stop.setEnabled(False)
        c_layout.addWidget(self.btn_stop)
        
        self.btn_reconnect = self.create_premium_button("RECONNECT FEED")
        self.btn_reconnect.clicked.connect(self.reconnect_camera)
        c_layout.addWidget(self.btn_reconnect)
        
        control_group.setLayout(c_layout)
        layout.addWidget(control_group)
        
        layout.addStretch()
        return widget
    
    def create_monitor_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(15, 20, 15, 20)
        layout.setSpacing(20)
        
        learn_group = self.create_card_group("SMART IDENTITY LOCK")
        learn_layout = QVBoxLayout()
        
        instruction = QLabel("Locks onto specific object IDs. Select multiple targets from the current frame.")
        instruction.setWordWrap(True)
        instruction.setStyleSheet(f"color: {THEME['text_dim']}; font-size: 11px; margin-bottom: 5px;")
        learn_layout.addWidget(instruction)
        
        self.btn_learn = self.create_premium_button("LOCK ON TARGET (LEARN)", primary=True)
        self.btn_learn.setStyleSheet(self.btn_learn.styleSheet() + f"border: 1px solid {THEME['accent_primary']};")
        self.btn_learn.clicked.connect(self.start_learning_sequence)
        learn_layout.addWidget(self.btn_learn)
        
        learn_group.setLayout(learn_layout)
        layout.addWidget(learn_group)
        
        manual_group = self.create_card_group("GENERAL CLASS MONITORING")
        m_layout = QVBoxLayout()
        
        self.object_combo = QComboBox()
        self.object_combo.addItems(sorted(self.all_objects))
        self.object_combo.setStyleSheet(self.get_combo_style())
        m_layout.addWidget(self.object_combo)
        
        action_row = QHBoxLayout()
        btn_add = self.create_premium_button("ADD CLASS")
        btn_add.clicked.connect(self.add_class_to_monitor)
        
        action_row.addWidget(btn_add)
        m_layout.addLayout(action_row)
        
        manual_group.setLayout(m_layout)
        layout.addWidget(manual_group)
        
        self.monitored_list = QListWidget()
        self.monitored_list.setStyleSheet(f"""
            QListWidget {{
                background-color: {THEME['bg_dark']};
                border: 1px solid {THEME['border']};
                border-radius: 4px;
                color: {THEME['text_light']};
                font-family: monospace;
            }}
            QListWidget::item {{
                padding: 10px;
                border-bottom: 1px solid {THEME['border']};
            }}
            QListWidget::item:selected {{
                background-color: {THEME['accent_primary']}40;
                border-left: 3px solid {THEME['accent_primary']};
            }}
        """)
        
        btn_remove = self.create_premium_button("REMOVE SELECTED")
        btn_remove.clicked.connect(self.remove_from_monitor)
        
        layout.addWidget(self.monitored_list)
        layout.addWidget(btn_remove)
        
        return widget
    
    def create_settings_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(15, 20, 15, 20)
        
        det_group = self.create_card_group("DETECTION THRESHOLD")
        d_layout = QVBoxLayout()
        
        self.conf_val_label = QLabel("50%")
        self.conf_val_label.setAlignment(Qt.AlignCenter)
        self.conf_val_label.setStyleSheet(f"color: {THEME['accent_primary']}; font-weight: bold; font-size: 16px;")
        d_layout.addWidget(self.conf_val_label)
        
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(10, 95)
        self.threshold_slider.setValue(50)
        self.threshold_slider.valueChanged.connect(self.update_threshold)
        d_layout.addWidget(self.threshold_slider)
        
        det_group.setLayout(d_layout)
        layout.addWidget(det_group)
        
        vis_group = self.create_card_group("DISPLAY OVERLAYS")
        v_layout = QVBoxLayout()
        
        self.chk_labels = QCheckBox("Show Labels & IDs")
        self.chk_labels.setChecked(True)
        self.chk_labels.stateChanged.connect(lambda: setattr(self.video_thread, 'show_labels', self.chk_labels.isChecked()))
        
        self.chk_conf = QCheckBox("Show Confidence Scores")
        self.chk_conf.setChecked(True)
        self.chk_conf.stateChanged.connect(lambda: setattr(self.video_thread, 'show_confidence', self.chk_conf.isChecked()))
        
        for chk in [self.chk_labels, self.chk_conf]:
            chk.setStyleSheet(f"color: {THEME['text_light']}; spacing: 10px;")
            v_layout.addWidget(chk)
            
        vis_group.setLayout(v_layout)
        layout.addWidget(vis_group)
        
        layout.addStretch()
        return widget
    
    def create_video_display(self):
        panel = QFrame()
        panel.setStyleSheet(f"""
            QFrame {{
                background-color: {THEME['bg_dark']};
                border: 2px solid {THEME['border']};
                border-radius: 4px;
            }}
        """)
        
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        
        hud = QFrame()
        hud.setFixedHeight(40)
        hud.setStyleSheet(f"background-color: {THEME['bg_panel']}; border-radius: 2px;")
        hud_layout = QHBoxLayout(hud)
        hud_layout.setContentsMargins(10, 0, 10, 0)
        
        self.source_indicator = QLabel("NO SIGNAL")
        self.fps_indicator = QLabel("0 FPS")
        self.obj_indicator = QLabel("0 OBJECTS")
        
        for lbl in [self.source_indicator, self.fps_indicator, self.obj_indicator]:
            lbl.setStyleSheet(f"color: {THEME['accent_primary']}; font-family: monospace; font-weight: bold;")
            hud_layout.addWidget(lbl)
        
        hud_layout.addStretch()
        layout.addWidget(hud)
        
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(800, 600)
        self.video_label.setStyleSheet(f"color: {THEME['text_dim']}; font-size: 14px;")
        self.video_label.setText("INITIALIZE SURVEILLANCE SYSTEM\nTO BEGIN FEED")
        
        layout.addWidget(self.video_label)
        return panel
    
    def create_info_panel(self):
        panel = QFrame()
        panel.setFixedWidth(350)
        panel.setStyleSheet(f"background-color: {THEME['bg_panel']}; border: 1px solid {THEME['border']};")
        
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(15, 15, 15, 15)
        
        stats_group = self.create_card_group("REAL-TIME METRICS")
        s_layout = QVBoxLayout()
        
        self.stat_total = self.create_stat_card("TOTAL TRACKED", "0")
        self.stat_monitored = self.create_stat_card("WATCHLIST", "0")
        self.stat_alerts = self.create_stat_card("ALERTS", "0", color=THEME['danger'])
        
        s_layout.addWidget(self.stat_total)
        s_layout.addWidget(self.stat_monitored)
        s_layout.addWidget(self.stat_alerts)
        stats_group.setLayout(s_layout)
        layout.addWidget(stats_group)
        
        log_group = self.create_card_group("SYSTEM LOGS")
        l_layout = QVBoxLayout()
        
        self.alert_text = QTextEdit()
        self.alert_text.setReadOnly(True)
        self.alert_text.setStyleSheet(f"""
            background-color: {THEME['bg_dark']};
            border: 1px solid {THEME['border']};
            color: {THEME['text_dim']};
            font-family: monospace;
            font-size: 11px;
        """)
        l_layout.addWidget(self.alert_text)
        
        btn_clear = self.create_premium_button("CLEAR LOGS")
        btn_clear.clicked.connect(self.clear_alert_log)
        l_layout.addWidget(btn_clear)
        
        log_group.setLayout(l_layout)
        layout.addWidget(log_group)
        
        return panel
    
    def create_status_bar(self):
        bar = QFrame()
        bar.setFixedHeight(35)
        bar.setStyleSheet(f"background-color: {THEME['bg_panel']}; border-top: 1px solid {THEME['accent_primary']};")
        
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(15, 0, 15, 0)
        
        self.status_label = QLabel("READY")
        self.status_label.setStyleSheet(f"color: {THEME['text_light']}; font-weight: bold;")
        
        self.time_label = QLabel()
        self.time_label.setStyleSheet(f"color: {THEME['text_dim']}; font-family: monospace;")
        
        timer = QTimer(self)
        timer.timeout.connect(self.update_time)
        timer.start(1000)
        
        layout.addWidget(self.status_label)
        layout.addStretch()
        layout.addWidget(self.time_label)
        
        return bar
        
    def create_card_group(self, title):
        group = QGroupBox(title)
        group.setStyleSheet(f"""
            QGroupBox {{
                font-size: 12px;
                font-weight: bold;
                color: {THEME['text_light']};
                border: 1px solid {THEME['border']};
                border-radius: 4px;
                margin-top: 20px;
                padding-top: 15px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 10px;
                color: {THEME['accent_primary']};
                background-color: {THEME['bg_panel']};
                left: 10px;
            }}
        """)
        return group
    
    def create_stat_card(self, label, value, color=THEME['text_light']):
        frame = QFrame()
        frame.setStyleSheet(f"background-color: {THEME['bg_dark']}; border-radius: 4px;")
        layout = QHBoxLayout(frame)
        
        lbl = QLabel(label)
        lbl.setStyleSheet(f"color: {THEME['text_dim']}; font-size: 11px;")
        
        val = QLabel(value)
        val.setStyleSheet(f"color: {color}; font-size: 18px; font-weight: bold;")
        val.setAlignment(Qt.AlignRight)
        
        layout.addWidget(lbl)
        layout.addWidget(val)
        return frame
        
    def create_premium_button(self, text, primary=False, danger=False):
        btn = QPushButton(text)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setMinimumHeight(40)
        
        base_style = f"""
            QPushButton {{
                border-radius: 4px;
                font-weight: bold;
                font-size: 12px;
                letter-spacing: 1px;
                padding: 5px;
            }}
        """
        
        if primary:
            style = base_style + f"""
                QPushButton {{
                    background-color: {THEME['accent_primary']};
                    color: {THEME['bg_dark']};
                    border: none;
                }}
                QPushButton:hover {{ background-color: {THEME['accent_secondary']}; }}
            """
        elif danger:
            style = base_style + f"""
                QPushButton {{
                    background-color: transparent;
                    color: {THEME['danger']};
                    border: 1px solid {THEME['danger']};
                }}
                QPushButton:hover {{ background-color: {THEME['danger']}; color: {THEME['text_light']}; }}
            """
        else:
            style = base_style + f"""
                QPushButton {{
                    background-color: {THEME['bg_panel']};
                    color: {THEME['text_light']};
                    border: 1px solid {THEME['border']};
                }}
                QPushButton:hover {{ border: 1px solid {THEME['accent_primary']}; }}
            """
            
        btn.setStyleSheet(style)
        return btn
    
    def get_combo_style(self):
        return f"""
            QComboBox {{
                background-color: {THEME['bg_dark']};
                color: {THEME['text_light']};
                border: 1px solid {THEME['border']};
                padding: 8px;
            }}
            QComboBox::drop-down {{ border: none; }}
        """

    def browse_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Video", "", "Video Files (*.mp4 *.avi *.mkv)")
        if path:
            self.video_file = path
            self.file_label.setText(os.path.basename(path))
            self.file_label.setStyleSheet(f"color: {THEME['success']};")
            
    def start_learning_sequence(self):
        if not self.video_thread.running:
            self.show_message("System Offline", "Please start surveillance first.", "warning")
            return
        
        self.status_label.setText("LOCKING ON TARGET...")
        self.status_label.setStyleSheet(f"color: {THEME['accent_primary']}; font-weight: bold;")
        self.video_thread.start_learning_mode()
        
    def handle_learn_result(self, result):
        if result['status'] == "timeout":
            self.show_message("Scan Timeout", "No clear object detected to lock onto.", "warning")
            self.status_label.setText("READY")
        elif result['status'] == "candidates":
            candidates = result['candidates']
            if not candidates:
                self.show_message("No Targets", "No objects found in frame.", "warning")
                return
            
            self.open_selection_dialog(candidates)
            self.status_label.setText("MONITORING ACTIVE")

    def open_selection_dialog(self, candidates):
        dialog = QDialog(self)
        dialog.setWindowTitle("Select Targets")
        dialog.setMinimumWidth(400)
        dialog.setStyleSheet(f"background-color: {THEME['bg_panel']}; color: {THEME['text_light']};")
        
        layout = QVBoxLayout(dialog)
        label = QLabel("Select objects to track:")
        layout.addWidget(label)
        
        list_widget = QListWidget()
        list_widget.setStyleSheet(f"background-color: {THEME['bg_dark']}; border: 1px solid {THEME['border']};")
        
        items_map = {}
        for cand in candidates:
            cls = cand['class']
            oid = cand['id']
            id_str = f"ID: {oid}" if oid is not None else "No ID"
            conf = int(cand['conf'] * 100)
            
            text = f"{cls.upper()} ({id_str}) - {conf}% Conf"
            item = QListWidgetItem(text)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            
            list_widget.addItem(item)
            items_map[id(item)] = cand
            
        layout.addWidget(list_widget)
        
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        
        if dialog.exec_() == QDialog.Accepted:
            for i in range(list_widget.count()):
                item = list_widget.item(i)
                if item.checkState() == Qt.Checked:
                    cand = items_map[id(item)]
                    self.register_candidate(cand)

    def register_candidate(self, cand):
        obj_class = cand['class']
        obj_id = cand['id']
        obj_embed = cand.get('embedding')
        id_str = f"ID: {obj_id}" if obj_id is not None else "No ID"
        
        name, ok = QInputDialog.getText(self, "Name Target", 
                                      f"Assign label for {obj_class.upper()} ({id_str}):")
        if ok and name:
            if name in self.video_thread.monitored_objects:
                name = f"{name}_{random.randint(100,999)}"
                
            self.video_thread.add_monitored_object(name, obj_class, obj_id, obj_embed)
            
            info_text = f"{name.upper()} | {obj_class} #{obj_id}"
            item = QListWidgetItem(info_text)
            self.monitored_list.addItem(item)
            
    def add_class_to_monitor(self):
        obj = self.object_combo.currentText()
        name, ok = QInputDialog.getText(self, "General Monitor", f"Monitor ALL objects of type '{obj}'?\nEnter label:")
        if ok and name:
            self.video_thread.add_monitored_object(name, obj, None)
            self.monitored_list.addItem(f"{name.upper()} | ALL {obj.upper()}")
            
    def remove_from_monitor(self):
        row = self.monitored_list.currentRow()
        if row >= 0:
            text = self.monitored_list.item(row).text()
            name = text.split(" | ")[0].strip()
            self.video_thread.remove_monitored_object_by_name(name.lower())
            
            keys = list(self.video_thread.monitored_objects.keys())
            for k in keys:
                if k.upper() == name:
                    self.video_thread.remove_monitored_object_by_name(k)
            
            self.monitored_list.takeItem(row)
            
    def update_threshold(self):
        val = self.threshold_slider.value()
        self.conf_val_label.setText(f"{val}%")
        self.video_thread.detection_threshold = val / 100.0
        
    def start_monitoring(self):
        source_idx = self.source_combo.currentIndex()
        source = 0
        if source_idx == 1: source = 1
        elif source_idx == 2:
            if not hasattr(self, 'video_file'):
                self.show_message("Error", "No video file selected.", "error")
                return
            source = self.video_file
            
        if self.video_thread.set_video_source(source):
            self.video_thread.start()
            self.btn_start.setEnabled(False)
            self.btn_stop.setEnabled(True)
            self.source_indicator.setText("SIGNAL ACTIVE")
            self.source_indicator.setStyleSheet(f"color: {THEME['success']}; font-weight: bold;")
            self.system_status.setText("SYSTEM ACTIVE")
            self.system_status.setStyleSheet(f"background-color: {THEME['success']}40; color: {THEME['success']}; border: 1px solid {THEME['success']}; padding: 8px 16px; border-radius: 4px;")
        else:
            self.show_message("Error", "Could not access video source.", "error")
            
    def stop_monitoring(self):
        self.video_thread.stop()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.video_label.setText("SIGNAL TERMINATED")
        self.source_indicator.setText("NO SIGNAL")
        self.source_indicator.setStyleSheet(f"color: {THEME['danger']}; font-weight: bold;")
        self.system_status.setText("SYSTEM STANDBY")
        self.system_status.setStyleSheet(f"background-color: {THEME['bg_panel']}; color: {THEME['text_dim']}; border: 1px solid {THEME['text_dim']}; padding: 8px 16px; border-radius: 4px;")
        
    def reconnect_camera(self):
        if self.video_thread.running:
            self.stop_monitoring()
            QTimer.singleShot(1000, self.start_monitoring)
            
    def update_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pix = QPixmap.fromImage(img)
        self.video_label.setPixmap(pix.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        
    def update_statistics(self, stats):
        self.update_stat_card(self.stat_total, str(stats['total_objects']))
        self.obj_indicator.setText(f"{stats['total_objects']} OBJECTS")
        
        monitored_data = stats['monitored_data']
        self.update_stat_card(self.stat_monitored, str(len(monitored_data)))
        
    def update_stat_card(self, widget, value):
        layout = widget.layout()
        if layout.count() > 1:
            layout.itemAt(1).widget().setText(value)
            
    def handle_alert(self, msg, timestamp):
        self.alert_log.append((timestamp, msg))
        formatted = f"<span style='color:{THEME['text_dim']}'>[{timestamp}]</span> <span style='color:{THEME['danger']}'>{msg}</span>"
        self.alert_text.append(formatted)
        
        current_alerts = int(self.stat_alerts.layout().itemAt(1).widget().text())
        self.update_stat_card(self.stat_alerts, str(current_alerts + 1))
        
    def update_fps(self, fps):
        self.fps_indicator.setText(f"{fps} FPS")
        
    def clear_alert_log(self):
        self.alert_text.clear()
        self.update_stat_card(self.stat_alerts, "0")
        
    def update_time(self):
        self.time_label.setText(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
    def show_message(self, title, msg, icon_type):
        box = QMessageBox(self)
        box.setWindowTitle(title)
        box.setText(msg)
        box.setStyleSheet(f"background-color: {THEME['bg_panel']}; color: {THEME['text_light']};")
        if icon_type == "error": box.setIcon(QMessageBox.Critical)
        elif icon_type == "warning": box.setIcon(QMessageBox.Warning)
        else: box.setIcon(QMessageBox.Information)
        box.exec_()

    def closeEvent(self, event):
        self.video_thread.stop()
        event.accept()

def main():
    app = QApplication(sys.argv)
    
    font = QFont()
    font.setFamily(font.defaultFamily())
    app.setFont(font)
    
    app.setStyle('Fusion')
    
    window = SurveillanceSystem()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()