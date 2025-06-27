import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import logging
from deep_sentinel.utils import logging_utils

logger = logging_utils.setup_module_logger(__name__)

class EmailNotifier:
    """Sends email alerts using SMTP
    
    Attributes:
        smtp_server: SMTP server address
        smtp_port: SMTP server port
        username: SMTP username
        password: SMTP password
        sender: Sender email address
        recipients: List of recipient email addresses
    """
    
    def __init__(self, smtp_server, smtp_port, username, password, sender, recipients):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.sender = sender
        self.recipients = recipients
        logger.info("Email notifier initialized")
    
    def send_alert(self, threat, message, image=None):
        """
        Send threat alert email
        
        Args:
            threat: Threat dictionary
            message: Alert message text
            image: Optional threat image
        """
        # Create email container
        msg = MIMEMultipart()
        msg['Subject'] = f"DeepSentinel Alert: {threat['type']} detected"
        msg['From'] = self.sender
        msg['To'] = ", ".join(self.recipients)
        
        # Create HTML body
        html = f"""
        <html>
            <body>
                <h2>Security Alert</h2>
                <p>{message}</p>
                <p>Threat Type: {threat['type']}</p>
                <p>Confidence: {threat['confidence']*100:.1f}%</p>
                <p>Location: {threat['location']}</p>
                <p>Timestamp: {threat['timestamp']}</p>
                {self._get_image_html(image) if image else ''}
            </body>
        </html>
        """
        msg.attach(MIMEText(html, 'html'))
        
        # Attach image if available
        if image is not None:
            img_part = MIMEImage(image)
            img_part.add_header('Content-Disposition', 'attachment', filename='threat.jpg')
            msg.attach(img_part)
        
        # Send email
        try:
            with smtplib.SMTP_SSL(self.smtp_server, self.smtp_port) as server:
                server.login(self.username, self.password)
                server.sendmail(self.sender, self.recipients, msg.as_string())
            logger.info(f"Email alert sent to {len(self.recipients)} recipients")
            return True
        except Exception as e:
            logger.error(f"Email sending failed: {str(e)}")
            return False
    
    def _get_image_html(self, image):
        """Generate HTML for inline image"""
        return f'<img src="cid:threat_image" alt="Threat Image"><br>'