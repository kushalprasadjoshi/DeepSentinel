import cv2
import numpy as np
from deep_sentinel.utils import logging_utils, video_utils

logger = logging_utils.setup_module_logger(__name__)

class MotionAnalyzer:
    """Detects motion in video frames using background subtraction
    
    Attributes:
        background_subtractor: Background subtractor object
        min_area: Minimum contour area to consider as motion
        history_length: Number of frames for background model
        detect_shadows: Whether to detect and ignore shadows
        sensitivity: Motion detection sensitivity (0-100)
    """
    
    def __init__(self, config):
        """
        Initialize motion analyzer
        
        Args:
            config: Application configuration dictionary
        """
        motion_config = config['detection']['motion']
        
        # Background subtraction parameters
        self.history_length = motion_config.get('history', 500)
        self.detect_shadows = motion_config.get('detect_shadows', True)
        self.sensitivity = motion_config.get('sensitivity', 50)
        self.min_area = motion_config.get('min_area', 500)
        
        # Create background subtractor
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self.history_length,
            detectShadows=self.detect_shadows
        )
        
        # Set initial background
        self.background = None
        logger.info("Motion analyzer initialized")
    
    def detect_motion(self, frame):
        """
        Detect motion in a frame using background subtraction
        
        Args:
            frame: Input frame (BGR or grayscale)
            
        Returns:
            bool: True if significant motion detected
            frame: Motion mask visualization (optional)
        """
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # Apply background subtraction
        fg_mask = self.background_subtractor.apply(gray)
        
        # Apply threshold based on sensitivity
        threshold = int(255 * (self.sensitivity / 100))
        _, thresh = cv2.threshold(fg_mask, threshold, 255, cv2.THRESH_BINARY)
        
        # Apply morphological operations to reduce noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.dilate(cleaned, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Check for significant motion
        motion_detected = False
        for contour in contours:
            if cv2.contourArea(contour) > self.min_area:
                motion_detected = True
                break
        
        return motion_detected
    
    def update_background(self):
        """Update background model manually"""
        self.background = self.background_subtractor.getBackgroundImage()
    
    def get_background(self):
        """Get current background model"""
        if self.background is None:
            self.update_background()
        return self.background
    
    def frame_difference(self, frame1, frame2):
        """
        Detect motion using simple frame differencing
        
        Args:
            frame1: Previous frame
            frame2: Current frame
            
        Returns:
            bool: True if significant motion detected
            frame: Motion visualization frame
        """
        return video_utils.calculate_frame_difference(frame1, frame2, self.min_area)