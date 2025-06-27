import cv2
from deep_sentinel.utils import logging_utils

logger = logging_utils.setup_module_logger(__name__)

class CameraManager:
    """Manages camera sources and frame acquisition
    
    Attributes:
        camera_sources (dict): Available camera sources
        current_camera (int/str): Current active camera
        cap (cv2.VideoCapture): Current capture object
        config (dict): Camera configuration
    """
    
    def __init__(self, config):
        """
        Initialize camera manager
        
        Args:
            config: Camera configuration dictionary
        """
        self.config = config
        self.camera_sources = {}
        self.current_camera = None
        self.cap = None
        self.discover_cameras()
        
    def discover_cameras(self):
        """Discover available camera sources"""
        # Check local cameras (index-based)
        max_check = 5
        for i in range(max_check):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                self.camera_sources[i] = f"Camera {i}"
                cap.release()
                logger.info(f"Discovered local camera: {i}")
        
        # Add any configured network cameras
        if 'network_cameras' in self.config:
            for name, url in self.config['network_cameras'].items():
                self.camera_sources[url] = f"Network: {name}"
                logger.info(f"Added network camera: {name} ({url})")
        
        # Set default camera
        default_cam = self.config.get('default_index', 0)
        if default_cam in self.camera_sources:
            self.set_camera(default_cam)
        elif self.camera_sources:
            first_cam = list(self.camera_sources.keys())[0]
            self.set_camera(first_cam)
        else:
            logger.warning("No cameras discovered!")
    
    def set_camera(self, camera_id):
        """
        Set active camera
        
        Args:
            camera_id: Camera index or URL
        """
        if camera_id not in self.camera_sources:
            logger.error(f"Camera {camera_id} not available")
            return False
        
        # Release existing capture
        if self.cap and self.cap.isOpened():
            self.cap.release()
        
        # Create new capture
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            logger.error(f"Failed to open camera: {camera_id}")
            return False
        
        # Apply configuration
        self.apply_config()
        self.current_camera = camera_id
        logger.info(f"Switched to camera: {self.camera_sources[camera_id]}")
        return True
    
    def apply_config(self):
        """Apply camera configuration to current capture"""
        if not self.cap or not self.cap.isOpened():
            return
        
        # Set resolution
        width = self.config.get('width', 1280)
        height = self.config.get('height', 720)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        # Set FPS if available
        if 'fps' in self.config:
            self.cap.set(cv2.CAP_PROP_FPS, self.config['fps'])
    
    def get_frame(self):
        """
        Get current frame from active camera
        
        Returns:
            frame: Current video frame, or None if unavailable
        """
        if not self.cap or not self.cap.isOpened():
            return None
            
        ret, frame = self.cap.read()
        if ret:
            return frame
        return None
    
    def get_available_cameras(self):
        """
        Get list of available cameras
        
        Returns:
            list: [(camera_id, camera_name)]
        """
        return [(id, name) for id, name in self.camera_sources.items()]
    
    def release(self):
        """Release camera resources"""
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.cap = None
        logger.info("Camera resources released")