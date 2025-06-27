"""
DeepSentinel Core Module
========================

This package contains the core functionality of the DeepSentinel security system,
including processing pipelines and system management components.

Submodules:
- pipelines: Video and alert processing workflows
- system: Camera and state management systems
"""

from .system.camera_manager import CameraManager
from .system.state_manager import SystemState
from .pipelines.video_pipeline import VideoPipeline
from .pipelines.alert_pipeline import AlertPipeline

# Public API
__all__ = [
    'CameraManager',
    'SystemState',
    'VideoPipeline',
    'AlertPipeline'
]