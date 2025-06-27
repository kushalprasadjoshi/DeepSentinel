import cv2
import numpy as np
from datetime import datetime

def resize_frame(frame, max_dim=800):
    """
    Resize frame while maintaining aspect ratio
    
    Args:
        frame: Input frame (numpy array)
        max_dim: Maximum dimension (width or height)
    
    Returns:
        Resized frame
    """
    h, w = frame.shape[:2]
    if max(h, w) <= max_dim:
        return frame
        
    if w > h:
        new_w = max_dim
        new_h = int(h * (max_dim / w))
    else:
        new_h = max_dim
        new_w = int(w * (max_dim / h))
        
    return cv2.resize(frame, (new_w, new_h))

def draw_bounding_boxes(frame, detections, threat_colors):
    """
    Draw bounding boxes and labels on frame
    
    Args:
        frame: Input frame
        detections: List of detection dictionaries with keys:
            - 'bbox': [x1, y1, x2, y2]
            - 'type': threat type string
            - 'confidence': confidence score
        threat_colors: Dictionary mapping threat types to BGR colors
    
    Returns:
        Annotated frame
    """
    for detection in detections:
        x1, y1, x2, y2 = detection['bbox']
        threat_type = detection['type']
        confidence = detection['confidence']
        
        # Get color for this threat type
        color = threat_colors.get(threat_type, (0, 255, 0))  # Default green
        
        # Draw bounding box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        
        # Create label text
        label = f"{threat_type} {confidence:.2f}"
        
        # Calculate text size
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        
        # Draw label background
        cv2.rectangle(frame, (int(x1), int(y1) - h - 10), 
                     (int(x1) + w, int(y1)), color, -1)
        
        # Draw label text
        cv2.putText(frame, label, (int(x1), int(y1) - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return frame

def add_timestamp(frame):
    """
    Add current timestamp to frame
    
    Args:
        frame: Input frame
    
    Returns:
        Frame with timestamp overlay
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, timestamp, (10, frame.shape[0] - 10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return frame

def frame_to_bytes(frame, quality=95):
    """
    Convert frame to JPEG bytes for streaming
    
    Args:
        frame: Input frame
        quality: JPEG quality (0-100)
    
    Returns:
        bytes: JPEG-encoded frame
    """
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, buffer = cv2.imencode('.jpg', frame, encode_params)
    return buffer.tobytes()

def calculate_frame_difference(frame1, frame2, min_contour_area=500):
    """
    Calculate motion between two frames
    
    Args:
        frame1: First frame (grayscale)
        frame2: Second frame (grayscale)
        min_contour_area: Minimum contour area to consider
    
    Returns:
        bool: True if significant motion detected
        frame: Motion visualization frame
    """
    # Convert to grayscale if needed
    if len(frame1.shape) == 3:
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    if len(frame2.shape) == 3:
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # Compute absolute difference
    diff = cv2.absdiff(frame1, frame2)
    
    # Apply threshold
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    
    # Dilate to fill holes
    dilated = cv2.dilate(thresh, None, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Check for significant motion
    motion_detected = False
    motion_frame = cv2.cvtColor(frame2, cv2.COLOR_GRAY2BGR)
    
    for contour in contours:
        if cv2.contourArea(contour) > min_contour_area:
            motion_detected = True
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(motion_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
    
    return motion_detected, motion_frame