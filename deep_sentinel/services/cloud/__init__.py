"""
DeepSentinel Cloud Services
===========================

This subpackage contains cloud platform integrations for DeepSentinel.

Modules:
- aws_client: AWS service integration (S3, Rekognition, SNS)
- model_updater: Model over-the-air update service
"""

from .aws_client import AWSClient
from .model_updater import ModelUpdater

__all__ = [
    'AWSClient',
    'ModelUpdater'
]