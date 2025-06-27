import face_recognition
import cv2
import numpy as np
import os
import pickle
from deep_sentinel.utils import logging_utils

logger = logging_utils.setup_module_logger(__name__)

class FaceRecognizer:
    """Recognizes faces using deep learning models
    
    Attributes:
        known_face_encodings: List of known face encodings
        known_face_names: List of names corresponding to encodings
        tolerance: Recognition tolerance (lower = stricter)
        model: Face recognition model (hog or cnn)
    """
    
    def __init__(self, known_faces_dir='data/processed/faces', tolerance=0.6, model='hog'):
        """
        Initialize face recognizer
        
        Args:
            known_faces_dir: Directory with known face images
            tolerance: Recognition tolerance (0.4-0.6 recommended)
            model: 'hog' for CPU, 'cnn' for GPU acceleration
        """
        self.tolerance = tolerance
        self.model = model
        self.known_face_encodings = []
        self.known_face_names = []
        
        # Load known faces
        self.load_known_faces(known_faces_dir)
        logger.info(f"Face recognizer initialized with {len(self.known_face_encodings)} known faces")
    
    def load_known_faces(self, faces_dir):
        """Load known faces from directory"""
        if not os.path.exists(faces_dir):
            logger.warning(f"Known faces directory not found: {faces_dir}")
            return
            
        # Try to load precomputed encodings
        cache_file = os.path.join(faces_dir, 'face_encodings.pkl')
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                    self.known_face_encodings = data['encodings']
                    self.known_face_names = data['names']
                logger.info(f"Loaded {len(self.known_face_encodings)} face encodings from cache")
                return
            except Exception as e:
                logger.error(f"Failed to load face cache: {str(e)}")
        
        # Process images if cache not available
        for filename in os.listdir(faces_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(faces_dir, filename)
                image = face_recognition.load_image_file(image_path)
                
                # Detect face locations
                face_locations = face_recognition.face_locations(image, model=self.model)
                if not face_locations:
                    logger.warning(f"No faces found in {filename}")
                    continue
                
                # Get encodings for the first face found
                face_encodings = face_recognition.face_encodings(image, face_locations)
                if face_encodings:
                    self.known_face_encodings.append(face_encodings[0])
                    self.known_face_names.append(os.path.splitext(filename)[0])
        
        # Save to cache
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'encodings': self.known_face_encodings,
                    'names': self.known_face_names
                }, f)
            logger.info(f"Cached {len(self.known_face_encodings)} face encodings")
        except Exception as e:
            logger.error(f"Failed to cache face encodings: {str(e)}")
    
    def recognize_faces(self, frame):
        """
        Recognize faces in a frame
        
        Args:
            frame: Input image frame (BGR format)
            
        Returns:
            list: Recognized faces with:
                - name: Recognized name or 'Unknown'
                - location: (top, right, bottom, left)
                - distance: Recognition confidence distance
        """
        # Convert BGR to RGB (which face_recognition uses)
        rgb_frame = frame[:, :, ::-1]
        
        # Find all face locations and encodings
        face_locations = face_recognition.face_locations(rgb_frame, model=self.model)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        recognized_faces = []
        
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Compare with known faces
            distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(distances)
            
            name = "Unknown"
            distance = distances[best_match_index]
            
            if distance <= self.tolerance:
                name = self.known_face_names[best_match_index]
            
            recognized_faces.append({
                'name': name,
                'location': (top, right, bottom, left),
                'distance': distance
            })
        
        return recognized_faces
    
    def add_known_face(self, image, name):
        """
        Add a new known face to the database
        
        Args:
            image: Face image (numpy array)
            name: Name for the face
            
        Returns:
            bool: True if successfully added
        """
        try:
            # Convert BGR to RGB
            rgb_image = image[:, :, ::-1]
            
            # Detect face locations
            face_locations = face_recognition.face_locations(rgb_image, model=self.model)
            if not face_locations:
                logger.warning("No face found in image")
                return False
                
            # Get encoding for the first face
            face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
            if not face_encodings:
                logger.warning("Failed to encode face")
                return False
                
            # Add to known faces
            self.known_face_encodings.append(face_encodings[0])
            self.known_face_names.append(name)
            logger.info(f"Added new known face: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding face: {str(e)}")
            return False
    
    def save_known_faces(self, faces_dir='data/processed/faces'):
        """Save known faces to cache file"""
        cache_file = os.path.join(faces_dir, 'face_encodings.pkl')
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'encodings': self.known_face_encodings,
                    'names': self.known_face_names
                }, f)
            logger.info(f"Saved {len(self.known_face_encodings)} face encodings to cache")
            return True
        except Exception as e:
            logger.error(f"Failed to save face cache: {str(e)}")
            return False