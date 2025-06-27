"""
DeepSentinel Services Module
============================

This package contains integrations with external services for the DeepSentinel security system,
including alert notifications and cloud platform integrations.

Subpackages:
- alerts: Notification services (email, SMS)
- cloud: Cloud platform integrations (AWS, model updates)
"""

from .alerts.email_alert import EmailNotifier
from .alerts.sms_alert import SMSNotifier
from .cloud.aws_client import AWSClient
from .cloud.model_updater import ModelUpdater

# Public API
__all__ = [
    'EmailNotifier',
    'SMSNotifier',
    'AWSClient',
    'ModelUpdater'
]