import time
import threading
from deep_sentinel.utils import logging_utils

logger = logging_utils.setup_module_logger(__name__)

class SystemState:
    """Manages and tracks the system state
    
    Attributes:
        threat_events (list): List of detected threat events
        system_status (dict): Current system metrics
        config (dict): Current configuration
        alert_history (list): History of sent alerts
        camera_states (dict): State of each camera
        lock (threading.Lock): Thread safety lock
    """
    
    def __init__(self, initial_config):
        """
        Initialize system state
        
        Args:
            initial_config: Initial configuration dictionary
        """
        self.threat_events = []
        self.system_status = {
            'fps': 0,
            'cpu_usage': 0,
            'memory_usage': 0,
            'detection_count': 0,
            'last_update': time.time()
        }
        self.config = initial_config
        self.alert_history = []
        self.camera_states = {}
        self.lock = threading.Lock()
        logger.info("System state initialized")
    
    def add_threat_event(self, event):
        """
        Add a new threat event to the system state
        
        Args:
            event: Threat event dictionary with keys:
                - type: threat type
                - confidence: detection confidence
                - location: camera/source
                - timestamp: detection time
                - snapshot: frame snapshot
        """
        with self.lock:
            # Add to threat events
            self.threat_events.append(event)
            
            # Update detection count
            self.system_status['detection_count'] += 1
            self.system_status['last_update'] = time.time()
            
            logger.info(f"New threat event: {event['type']} at {event['location']}")
    
    def clear_threat_events(self, max_age=3600):
        """
        Clear old threat events
        
        Args:
            max_age: Maximum age in seconds to keep events
        """
        with self.lock:
            current_time = time.time()
            self.threat_events = [
                event for event in self.threat_events 
                if current_time - event['timestamp'] < max_age
            ]
    
    def update_system_status(self, status_update):
        """
        Update system status metrics
        
        Args:
            status_update: Dictionary of status values to update
        """
        with self.lock:
            self.system_status.update(status_update)
            self.system_status['last_update'] = time.time()
    
    def add_alert_history(self, alert):
        """
        Add an alert to the history
        
        Args:
            alert: Alert dictionary with keys:
                - type: threat type
                - timestamp: alert time
                - channels: delivery channels
                - recipients: who was notified
        """
        with self.lock:
            self.alert_history.append(alert)
            logger.info(f"Alert sent: {alert['type']} via {alert['channels']}")
    
    def update_camera_state(self, camera_id, state):
        """
        Update state for a specific camera
        
        Args:
            camera_id: Camera identifier
            state: Dictionary of camera state values
        """
        with self.lock:
            if camera_id not in self.camera_states:
                self.camera_states[camera_id] = {}
            self.camera_states[camera_id].update(state)
    
    def get_config(self):
        """Get current configuration"""
        return self.config.copy()
    
    def update_config(self, new_config):
        """Update system configuration"""
        with self.lock:
            self.config.update(new_config)
            logger.info("Configuration updated")
    
    def get_recent_threats(self, count=10):
        """Get most recent threats"""
        with self.lock:
            return sorted(
                self.threat_events, 
                key=lambda x: x['timestamp'], 
                reverse=True
            )[:count]
    
    def get_current_threats(self):
        """Get threats since last check"""
        # This would be implemented to return new threats
        # since the last time this was called
        return []