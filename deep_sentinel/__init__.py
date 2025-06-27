"""
DeepSentinel - AI-Powered Security Surveillance System
======================================================

This package provides the core functionality for the DeepSentinel security system.
"""

__version__ = "1.0.0"

# Initialize package logging
import logging
import os

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Set up top-level logger
logger = logging.getLogger("deep_sentinel")
logger.setLevel(logging.INFO)

# Create file handler
file_handler = logging.FileHandler("logs/deepsentinel.log")
file_handler.setLevel(logging.INFO)

# Create console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Package initialization message
logger.info(f"Initializing DeepSentinel v{__version__}")

# Import core components
from .core.system.camera_manager import CameraManager
from .core.system.state_manager import SystemState
from .core.pipelines.video_pipeline import VideoProcessor
from .core.pipelines.alert_pipeline import AlertHandler

# Import AI components
from .ai.detection.object_detector import ThreatDetector
from .ai.detection.motion_detector import MotionAnalyzer
from .ai.detection.face_recognizer import FaceRecognizer
from .ai.training.trainer import ModelTrainer
from .ai.training.validator import ModelValidator
from .ai.training.augmentor import DataAugmentor

# Import interface components
from .interfaces.gui.main_window import MainApplication
from .interfaces.gui.dashboard import ThreatDashboard
from .interfaces.gui.controls import ControlPanel
from .interfaces.voice.controller import VoiceController
from .interfaces.voice.synthesizer import VoiceSynthesizer

# Import service components
from .services.alerts.email_alert import EmailNotifier
from .services.alerts.sms_alert import SMSNotifier
from .services.cloud.aws_client import AWSClient
from .services.cloud.model_updater import ModelUpdater

# Import utility components
from .utils.config_loader import load_config
from .utils.video_utils import resize_frame, draw_bounding_boxes
from .utils.logging_utils import setup_module_logger

# Define public API
__all__ = [
    # Core components
    'CameraManager', 'SystemState', 'VideoProcessor', 'AlertHandler',
    
    # AI components
    'ThreatDetector', 'MotionAnalyzer', 'FaceRecognizer', 
    'ModelTrainer', 'ModelValidator', 'DataAugmentor',
    
    # Interface components
    'MainApplication', 'ThreatDashboard', 'ControlPanel',
    'VoiceController', 'VoiceSynthesizer',
    
    # Service components
    'EmailNotifier', 'SMSNotifier', 'AWSClient', 'ModelUpdater',
    
    # Utility components
    'load_config', 'resize_frame', 'draw_bounding_boxes', 'setup_module_logger'
]