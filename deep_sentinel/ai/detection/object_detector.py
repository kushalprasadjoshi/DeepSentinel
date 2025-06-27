import cv2
import numpy as np
from ultralytics import YOLO
from deep_sentinel.utils import logging_utils

logger = logging_utils.setup_module_logger(__name__)

class ThreatDetector:
    """Detects security threats using YOLOv8 model
    
    Attributes:
        model: YOLOv8 model instance
        classes: List of threat classes
        confidence_thresh: Minimum confidence threshold
        nms_thresh: Non-maximum suppression threshold
        threat_colors: Color mapping for threat types
    """
    
    def __init__(self, config):
        """
        Initialize threat detector
        
        Args:
            config: Application configuration dictionary
        """
        model_path = config['ai']['model_path']
        self.model = YOLO(model_path)
        self.classes = config['detection']['classes']
        self.confidence_thresh = config['detection']['confidence_threshold']
        self.nms_thresh = config['detection']['nms_threshold']
        
        # Create color map for threat types
        self.threat_colors = {}
        for threat_type, color in config['ui']['threat_colors'].items():
            self.threat_colors[threat_type] = tuple(color)
        
        logger.info(f"Loaded threat detection model from {model_path}")
    
    def detect(self, frame):
        """
        Detect threats in a frame
        
        Args:
            frame: Input image frame
            
        Returns:
            list: Detected threat objects with:
                - type: threat class
                - confidence: detection confidence
                - bbox: [x1, y1, x2, y2]
        """
        # Run inference
        results = self.model.predict(
            frame, 
            conf=self.confidence_thresh,
            iou=self.nms_thresh,
            verbose=False
        )
        
        detections = []
        
        for result in results:
            for box in result.boxes:
                # Get class ID and confidence
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])
                
                # Get class name
                threat_type = self.classes[cls_id]
                
                # Get bounding box coordinates
                bbox = box.xyxy[0].tolist()
                
                detections.append({
                    'type': threat_type,
                    'confidence': confidence,
                    'bbox': bbox
                })
        
        logger.debug(f"Detected {len(detections)} threats")
        return detections