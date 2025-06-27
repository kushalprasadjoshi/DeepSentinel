"""
DeepSentinel Utilities Module
=============================

This package contains utility functions and helper classes used throughout the DeepSentinel system.

Modules:
- config_loader: Configuration loading and management
- logging_utils: Logging setup and management
- video_utils: Video processing utilities
"""

from .config_loader import load_config
from .logging_utils import setup_logger, setup_module_logger
from .video_utils import (
    resize_frame,
    draw_bounding_boxes,
    add_timestamp,
    frame_to_bytes,
    calculate_frame_difference
)

# Public API
__all__ = [
    'load_config',
    'setup_logger',
    'setup_module_logger',
    'resize_frame',
    'draw_bounding_boxes',
    'add_timestamp',
    'frame_to_bytes',
    'calculate_frame_difference'
]