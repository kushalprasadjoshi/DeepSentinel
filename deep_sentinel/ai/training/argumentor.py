import albumentations as A
import cv2
import numpy as np
from deep_sentinel.utils import logging_utils

logger = logging_utils.setup_module_logger(__name__)

class DataAugmentor:
    """Handles data augmentation for security threat detection
    
    Attributes:
        augmentations: Albumentations augmentation pipeline
        bbox_params: Bounding box parameters for augmentation
    """
    
    def __init__(self, config):
        """
        Initialize data augmentor
        
        Args:
            config: Augmentation configuration from model_config.yaml
        """
        self.augmentations = self.build_augmentation_pipeline(config)
        self.bbox_params = A.BboxParams(
            format='yolo', 
            min_visibility=0.25,  # Minimum visibility after augmentation
            label_fields=['class_labels']
        )
        logger.info("Data augmentor initialized")

    def build_augmentation_pipeline(self, config):
        """
        Build augmentation pipeline from configuration
        
        Args:
            config: Augmentation configuration section
            
        Returns:
            Albumentations augmentation pipeline
        """
        aug_config = config.get('augmentation', {})
        
        # Base augmentations (always applied)
        transforms = [
            # Geometric transformations
            A.Affine(
                rotate=(-aug_config.get('degrees', 0), aug_config.get('degrees', 0)),
                translate_percent=(0, aug_config.get('translate', 0.1)),
                scale=(1 - aug_config.get('scale', 0.5), 1 + aug_config.get('scale', 0.5)),
                shear=(-aug_config.get('shear', 0), aug_config.get('shear', 0)),
                p=0.7
            ),
            
            # Color transformations
            A.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.1,
                hue=aug_config.get('hsv_h', 0.015),
                p=0.7
            ),
            
            # Weather effects
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, alpha_coef=0.08, p=0.1),
            A.RandomShadow(num_shadows_lower=1, num_shadows_upper=2, p=0.1),
            A.RandomRain(p=0.1),
        ]
        
        # Optional advanced augmentations
        if aug_config.get('mosaic', 1.0) > 0:
            transforms.append(A.RandomGridShuffle(grid=(2, 2), p=aug_config['mosaic']))
        
        if aug_config.get('mixup', 0.0) > 0:
            transforms.append(
                A.RandomBrightnessContrast(
                    brightness_limit=(-0.2, 0.2),
                    contrast_limit=(-0.2, 0.2),
                    p=aug_config['mixup']
                )
            )
        
        # Always include horizontal flip
        transforms.append(A.HorizontalFlip(p=aug_config.get('fliplr', 0.5)))
        
        # Compose the pipeline
        return A.Compose(transforms, bbox_params=self.bbox_params)

    def augment(self, image, bboxes, class_labels):
        """
        Apply augmentations to an image and its bounding boxes
        
        Args:
            image: Input image (numpy array)
            bboxes: List of bounding boxes in YOLO format [x_center, y_center, width, height]
            class_labels: List of class labels for each bounding box
            
        Returns:
            tuple: (augmented_image, augmented_bboxes, augmented_labels)
        """
        try:
            # Convert YOLO bboxes to Albumentations format [x_min, y_min, x_max, y_max]
            albu_bboxes = []
            for bbox in bboxes:
                x, y, w, h = bbox
                x_min = x - w/2
                y_min = y - h/2
                x_max = x + w/2
                y_max = y + h/2
                albu_bboxes.append([x_min, y_min, x_max, y_max])
            
            # Apply augmentations
            augmented = self.augmentations(
                image=image,
                bboxes=albu_bboxes,
                class_labels=class_labels
            )
            
            # Convert back to YOLO format
            yolo_bboxes = []
            for bbox in augmented['bboxes']:
                x_min, y_min, x_max, y_max = bbox
                w = x_max - x_min
                h = y_max - y_min
                x = x_min + w/2
                y = y_min + h/2
                yolo_bboxes.append([x, y, w, h])
                
            return augmented['image'], yolo_bboxes, augmented['class_labels']
        
        except Exception as e:
            logger.error(f"Augmentation failed: {str(e)}")
            return image, bboxes, class_labels

    def augment_security(self, image, bboxes, class_labels):
        """
        Apply security-specific augmentations to simulate challenging conditions
        
        Args:
            image: Input image
            bboxes: Bounding boxes
            class_labels: Class labels
            
        Returns:
            tuple: (augmented_image, augmented_bboxes, augmented_labels)
        """
        # Security-specific augmentations
        security_transforms = A.Compose([
            # Low-light conditions
            A.RandomGamma(gamma_limit=(60, 120), p=0.3),
            
            # Motion blur (simulating camera movement)
            A.MotionBlur(blur_limit=(3, 7), p=0.2),
            
            # Rain effects (for outdoor cameras)
            A.RandomRain(
                slant_lower=-5, 
                slant_upper=5, 
                drop_length=15, 
                drop_width=1, 
                p=0.2
            ),
            
            # Camera noise
            A.ISONoise(
                color_shift=(0.01, 0.05),
                intensity=(0.1, 0.5),
                p=0.2
            ),
            
            # Partial occlusions
            A.CoarseDropout(
                max_holes=8,
                max_height=32,
                max_width=32,
                min_holes=1,
                min_height=8,
                min_width=8,
                p=0.3
            )
        ], bbox_params=self.bbox_params)
        
        try:
            # Apply augmentations
            augmented = security_transforms(
                image=image,
                bboxes=bboxes,
                class_labels=class_labels
            )
            return augmented['image'], augmented['bboxes'], augmented['class_labels']
        except Exception as e:
            logger.error(f"Security augmentation failed: {str(e)}")
            return image, bboxes, class_labels

    def visualize_augmentation(self, image, bboxes, class_labels, output_path):
        """
        Visualize augmentation results (for debugging)
        
        Args:
            image: Original image
            bboxes: Original bounding boxes
            class_labels: Original class labels
            output_path: Path to save visualization
        """
        # Apply augmentation
        aug_img, aug_bboxes, aug_labels = self.augment(image.copy(), bboxes.copy(), class_labels.copy())
        
        # Draw original bounding boxes
        orig_img = image.copy()
        for bbox, label in zip(bboxes, class_labels):
            x, y, w, h = bbox
            x_min = int((x - w/2) * orig_img.shape[1])
            y_min = int((y - h/2) * orig_img.shape[0])
            x_max = int((x + w/2) * orig_img.shape[1])
            y_max = int((y + h/2) * orig_img.shape[0])
            cv2.rectangle(orig_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(orig_img, str(label), (x_min, y_min-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        
        # Draw augmented bounding boxes
        for bbox, label in zip(aug_bboxes, aug_labels):
            x, y, w, h = bbox
            x_min = int((x - w/2) * aug_img.shape[1])
            y_min = int((y - h/2) * aug_img.shape[0])
            x_max = int((x + w/2) * aug_img.shape[1])
            y_max = int((y + h/2) * aug_img.shape[0])
            cv2.rectangle(aug_img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
            cv2.putText(aug_img, str(label), (x_min, y_min-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        
        # Combine images
        combined = np.hstack([orig_img, aug_img])
        
        # Save result
        cv2.imwrite(output_path, combined)
        logger.info(f"Augmentation visualization saved to {output_path}")