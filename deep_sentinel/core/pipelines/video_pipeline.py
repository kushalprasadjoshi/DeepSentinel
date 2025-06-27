import time
import threading
import cv2
import numpy as np
from deep_sentinel.core.system import state_manager
from deep_sentinel.utils import video_utils, logging_utils
from deep_sentinel.ai.detection import object_detector, motion_detector

logger = logging_utils.setup_module_logger(__name__)

class VideoPipeline:
    """Handles the complete video processing pipeline
    
    Attributes:
        camera_manager: CameraManager instance
        state_manager: SystemState instance
        threat_detector: ThreatDetector instance
        motion_analyzer: MotionAnalyzer instance
        frame: Current processed frame
        running: Pipeline running status
        thread: Processing thread
        fps: Current frames per second
    """
    
    def __init__(self, camera_manager, state_manager, config):
        """
        Initialize video pipeline
        
        Args:
            camera_manager: CameraManager instance
            state_manager: SystemState instance
            config: Application configuration dictionary
        """
        self.camera_manager = camera_manager
        self.state_manager = state_manager
        self.threat_detector = object_detector.ThreatDetector(config)
        self.motion_analyzer = motion_detector.MotionAnalyzer(config)
        self.frame = None
        self.running = False
        self.thread = None
        self.fps = 0
        self.config = config
        self.prev_frame = None
        logger.info("Video pipeline initialized")
    
    def start(self):
        """Start video processing pipeline"""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._process)
            self.thread.daemon = True
            self.thread.start()
            logger.info("Video pipeline started")
    
    def stop(self):
        """Stop video processing pipeline"""
        if self.running:
            self.running = False
            if self.thread is not None:
                self.thread.join(timeout=2.0)
            logger.info("Video pipeline stopped")
    
    def _process(self):
        """Main processing loop"""
        frame_count = 0
        start_time = time.time()
        
        while self.running:
            # Get frame from camera
            raw_frame = self.camera_manager.get_frame()
            if raw_frame is None:
                time.sleep(0.1)
                continue
                
            # Preprocess frame
            frame = video_utils.resize_frame(raw_frame, max_dim=self.config['camera']['resolution']['width'])
            
            # Convert to grayscale for motion detection
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Motion detection
            motion_detected = False
            if self.prev_frame is not None:
                motion_detected, motion_frame = video_utils.calculate_frame_difference(
                    self.prev_frame, gray_frame, 
                    min_contour_area=self.config['detection']['motion']['min_area']
                )
            
            self.prev_frame = gray_frame
            
            # Threat detection if motion detected
            detections = []
            if motion_detected:
                # Detect threats
                detections = self.threat_detector.detect(frame)
                
                # Add threat events to state
                for detection in detections:
                    threat_event = {
                        'type': detection['type'],
                        'confidence': detection['confidence'],
                        'location': self.camera_manager.current_camera,
                        'timestamp': time.time(),
                        'snapshot': frame
                    }
                    self.state_manager.add_threat_event(threat_event)
            
            # Annotate frame
            annotated_frame = video_utils.draw_bounding_boxes(
                frame, 
                detections, 
                self.config['ui']['threat_colors']
            )
            annotated_frame = video_utils.add_timestamp(annotated_frame)
            
            # Update frame for external access
            self.frame = annotated_frame
            
            # Calculate FPS
            frame_count += 1
            if frame_count >= 10:
                elapsed = time.time() - start_time
                self.fps = frame_count / elapsed
                frame_count = 0
                start_time = time.time()
                
                # Update system status
                self.state_manager.update_system_status({'fps': self.fps})
            
            # Throttle processing if needed
            target_fps = self.config['camera']['fps']
            if target_fps > 0:
                time.sleep(max(0, 1/target_fps - (time.time() - start_time)))
    
    def get_current_frame(self):
        """Get current processed frame"""
        return self.frame
    
    def get_fps(self):
        """Get current processing FPS"""
        return self.fps