"""
DeepSentinel Interfaces Module
==============================

This package contains the user interface components for the DeepSentinel security system,
including both graphical (GUI) and voice interfaces.

Subpackages:
- gui: Graphical user interface components
- voice: Voice command and response interfaces
"""

from .gui.main_window import MainApplication
from .gui.dashboard import ThreatDashboard
from .gui.controls import ControlPanel
from .voice.controller import VoiceController
from .voice.synthesizer import VoiceSynthesizer

# Public API
__all__ = [
    'MainApplication',
    'ThreatDashboard',
    'ControlPanel',
    'VoiceController',
    'VoiceSynthesizer'
]