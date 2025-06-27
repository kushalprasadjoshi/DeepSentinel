from twilio.rest import Client
from deep_sentinel.utils import logging_utils

logger = logging_utils.setup_module_logger(__name__)

class SMSNotifier:
    """Sends SMS alerts using Twilio
    
    Attributes:
        client: Twilio client instance
        from_number: Twilio phone number
        recipients: List of recipient phone numbers
    """
    
    def __init__(self, account_sid, auth_token, from_number, recipients):
        self.client = Client(account_sid, auth_token)
        self.from_number = from_number
        self.recipients = recipients
        logger.info("SMS notifier initialized")
    
    def send_alert(self, threat, message):
        """
        Send threat alert SMS
        
        Args:
            threat: Threat dictionary
            message: Alert message text
        """
        # Format message
        full_message = f"🚨 {message}\nThreat: {threat['type']}\nConfidence: {threat['confidence']*100:.1f}%\nLocation: {threat['location']}"
        
        # Send to each recipient
        success = True
        for to_number in self.recipients:
            try:
                self.client.messages.create(
                    body=full_message,
                    from_=self.from_number,
                    to=to_number
                )
                logger.info(f"SMS alert sent to {to_number}")
            except Exception as e:
                logger.error(f"SMS failed to {to_number}: {str(e)}")
                success = False
        
        return success