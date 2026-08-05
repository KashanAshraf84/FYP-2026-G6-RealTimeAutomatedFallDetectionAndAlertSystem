"""
Fall Detection System - Configuration
======================================
Central configuration for all system parameters.
"""

import os
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class PoseConfig:
    """MediaPipe Pose estimation configuration."""
    model_complexity: int = 1          # 0=lite, 1=full, 2=heavy
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    enable_segmentation: bool = False
    smooth_landmarks: bool = True

    # Key landmark indices (MediaPipe 33 landmarks)
    NOSE: int = 0
    LEFT_SHOULDER: int = 11
    RIGHT_SHOULDER: int = 12
    LEFT_HIP: int = 23
    RIGHT_HIP: int = 24
    LEFT_KNEE: int = 25
    RIGHT_KNEE: int = 26
    LEFT_ANKLE: int = 27
    RIGHT_ANKLE: int = 28
    LEFT_EAR: int = 7
    RIGHT_EAR: int = 8

    # Critical keypoints for fall detection (YOLOv8-Pose 17 landmarks)
    CRITICAL_KEYPOINTS: List[int] = field(default_factory=lambda: [
        0,   # Nose
        5,   # Left Shoulder
        6,   # Right Shoulder
        7,   # Left Elbow
        8,   # Right Elbow
        9,   # Left Wrist
        10,  # Right Wrist
        11,  # Left Hip
        12,  # Right Hip
        13,  # Left Knee
        14,  # Right Knee
        15,  # Left Ankle
        16,  # Right Ankle
    ])


@dataclass
class FeatureConfig:
    """Feature extraction configuration."""
    sequence_length: int = 30          # Number of frames in a temporal window
    stride: int = 5                     # Stride for sliding window
    num_keypoints: int = 13            # Number of critical keypoints used (YOLO subset)
    coords_per_keypoint: int = 3       # x, y, confidence
    num_raw_features: int = 39         # num_keypoints * coords_per_keypoint
    num_engineered_features: int = 12  # Hand-crafted features per frame

    # Thresholds for rule-based fall pre-screening
    vertical_speed_threshold: float = 0.15     # Rapid downward movement
    body_angle_threshold: float = 45.0         # Body tilt angle (degrees)
    aspect_ratio_threshold: float = 1.0        # Width/Height ratio
    stillness_threshold: float = 0.005         # Post-fall stillness
    stillness_duration_frames: int = 15        # Frames of stillness to confirm fall

    # Below the fall drop-speed threshold (20 px/frame) but still a sudden
    # jerk/lunge — flags "warning" for a single frame instead of waiting for
    # the 3-of-5 temporal smoothing consensus, so brief jerks stay visible.
    jerk_speed_threshold: float = 12.0


@dataclass
class ModelConfig:
    """Neural network model configuration."""
    # Architecture
    architecture: str = "cnn_lstm"     # Options: "lstm", "cnn_lstm", "transformer"
    input_size: int = 51               # raw features + engineered features per frame
    hidden_size: int = 128
    num_layers: int = 2
    num_classes: int = 3               # normal, warning, fall
    dropout: float = 0.3
    bidirectional: bool = True

    # CNN-LSTM specific
    cnn_filters: List[int] = field(default_factory=lambda: [64, 128])
    cnn_kernel_size: int = 3

    # Transformer specific
    num_heads: int = 4
    ff_dim: int = 256
    num_transformer_layers: int = 2

    # Training
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    epochs: int = 100
    patience: int = 15                 # Early stopping patience
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 7

    # Class weights for imbalanced data [normal, warning, fall]
    class_weights: List[float] = field(default_factory=lambda: [1.0, 2.0, 3.0])


@dataclass
class CameraConfig:
    """Camera / video input configuration."""
    source: int = 0                    # 0 = default webcam, or path to video file
    width: int = 640
    height: int = 480
    fps: int = 30
    buffer_size: int = 1


@dataclass
class AlertConfig:
    """Alert system configuration."""
    enable_sound: bool = True
    enable_email: bool = False
    enable_logging: bool = True

    # Email settings
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    sender_email: str = ""
    sender_password: str = ""
    recipient_emails: List[str] = field(default_factory=list)

    # Logging
    log_dir: str = "logs"
    alert_cooldown_seconds: float = 30.0   # Min time between alerts

    # Sound
    alarm_sound_path: str = "assets/alarm.wav"

    # Visual popup
    popup_duration_ms: int = 3000          # How long the on-screen alert popup stays visible

    # Deployment profile.
    # True on machines with no desktop session or audio device (e.g. a cloud
    # container). Suppresses the buzzer, the OpenCV popup and the OS
    # notification; console output and JSON logging are unaffected.
    headless: bool = False


@dataclass
class SystemConfig:
    """Top-level system configuration."""
    pose: PoseConfig = field(default_factory=PoseConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    alert: AlertConfig = field(default_factory=AlertConfig)

    # Paths
    project_dir: str = os.path.dirname(os.path.abspath(__file__))
    model_save_path: str = "models/fall_detector.pth"
    onnx_model_path: str = "models/fall_detector.onnx"
    yolo_model_path: str = "yolov8n-pose.pt"
    data_dir: str = "data"
    processed_data_dir: str = "data/processed"
    database_path: str = "data/guardianai.db"

    # Display
    show_pose: bool = True
    show_features: bool = True
    show_fps: bool = True
    window_name: str = "Fall Detection System"

    # System
    device: str = "auto"  # "auto", "cpu", "cuda"
    confidence_threshold: float = 0.7  # Min confidence to trigger status change

    def __post_init__(self):
        """Fix relative paths to be absolute based on project_dir."""
        # Fix Alert paths
        if not os.path.isabs(self.alert.log_dir):
            self.alert.log_dir = os.path.join(self.project_dir, self.alert.log_dir)
        if not os.path.isabs(self.alert.alarm_sound_path):
            self.alert.alarm_sound_path = os.path.join(self.project_dir, self.alert.alarm_sound_path)
            
        # Fix Model paths
        if not os.path.isabs(self.model_save_path):
            self.model_save_path = os.path.join(self.project_dir, self.model_save_path)
        if not os.path.isabs(self.onnx_model_path):
            self.onnx_model_path = os.path.join(self.project_dir, self.onnx_model_path)
            
        # Fix Data paths
        if not os.path.isabs(self.data_dir):
            self.data_dir = os.path.join(self.project_dir, self.data_dir)
        if not os.path.isabs(self.processed_data_dir):
            self.processed_data_dir = os.path.join(self.project_dir, self.processed_data_dir)
        if not os.path.isabs(self.database_path):
            self.database_path = os.path.join(self.project_dir, self.database_path)

    def get_device(self):
        """Get the appropriate torch device."""
        import torch
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device


# Global configuration instance
CONFIG = SystemConfig()
