import time
from deep_sentinel.services.alerts import email_alert, sms_alert
from deep_sentinel.utils import logging_utils

logger = logging_utils.setup_module_logger(__name__)

class AlertPipeline:
    """Handles alert generation and delivery
    
    Attributes:
        state_manager: SystemState instance
        email_notifier: EmailNotifier instance
        sms_notifier: SMSNotifier instance
        alert_cooldowns (dict): Cooldown timers for each alert type
        last_alert_time (float): Timestamp of last alert
    """
    
    def __init__(self, state_manager, email_notifier=None, sms_notifier=None):
        self.state_manager = state_manager
        self.email_notifier = email_notifier
        self.sms_notifier = sms_notifier
        self.alert_cooldowns = {}
        self.last_alert_time = 0
        logger.info("Alert pipeline initialized")
    
    def start_monitoring(self):
        """Start monitoring for threats and sending alerts"""
        # In a real implementation, this would run in a separate thread
        # For simplicity, we'll process in the main loop
        pass
    
    def process_new_threats(self):
        """Check for new threats and send alerts"""
        current_threats = self.state_manager.current_threats
        config = self.state_manager.get_config().get('alerts', {})
        
        for threat in current_threats:
            threat_type = threat['type']
            confidence = threat['confidence']
            
            # Check if this threat type has alert rules
            rules = config.get('rules', {}).get(threat_type, config.get('other', {}))
            min_confidence = rules.get('min_confidence', 0.8)
            cooldown = rules.get('cooldown', 300)  # default 5 minutes
            
            # Check confidence threshold
            if confidence < min_confidence:
                continue
                
            # Check cooldown
            current_time = time.time()
            last_alert_time = self.alert_cooldowns.get(threat_type, 0)
            if current_time - last_alert_time < cooldown:
                continue
                
            # Send alerts
            message = f"{threat_type} detected at {threat['location']} with {confidence*100:.1f}% confidence"
            channels = rules.get('channels', ['email'])
            
            if 'email' in channels and self.email_notifier:
                self.email_notifier.send_alert(threat, message)
            if 'sms' in channels and self.sms_notifier:
                self.sms_notifier.send_alert(threat, message)
            
            # Update cooldown
            self.alert_cooldowns[threat_type] = current_time
            self.last_alert_time = current_time
            logger.info(f"Sent {threat_type} alert via {channels}")
        
        # Clear current threats after processing
        self.state_manager.clear_current_threats()