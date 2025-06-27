"""
DeepSentinel - Core AI Security Package
=======================================

This package provides the core functionality for the DeepSentinel security system.
"""

__version__ = "1.0.0"

# Initialize package logging
import logging

# Set up top-level logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Package initialization message
logger.info(f"Initializing DeepSentinel v{__version__}")

# Import core components
from .core.processor import VideoProcessor
from .core.config_manager import ConfigManager
from .core.state_manager import SystemState

# Import AI components
from .ai.detection.object_detector import ThreatDetector
from .ai.detection.motion_detector import MotionAnalyzer

# Import interface components
from .interfaces.gui.main_window import MainApplication
from .interfaces.voice.controller import VoiceController

# Import service components
from .services.alerts.email_alert import EmailNotifier
from .services.alerts.sms_alert import SMSNotifier

# Utility imports
from .utils.config_loader import load_config
from .utils.video_utils import open_camera_source

# Define public API
__all__ = [
    'VideoProcessor',
    'ConfigManager',
    'SystemState',
    'ThreatDetector',
    'MotionAnalyzer',
    'MainApplication',
    'VoiceController',
    'EmailNotifier',
    'SMSNotifier',
    'load_config',
    'open_camera_source'
]