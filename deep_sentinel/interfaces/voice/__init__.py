"""
DeepSentinel Voice Interface
============================

This subpackage contains components for voice-controlled interaction with the DeepSentinel system.

Modules:
- controller: Voice command recognition and processing
- synthesizer: Text-to-speech output generation
"""

from .controller import VoiceController
from .synthesizer import VoiceSynthesizer

__all__ = [
    'VoiceController',
    'VoiceSynthesizer'
]