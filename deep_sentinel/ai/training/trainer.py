import os
import yaml
import time
import torch
from ultralytics import YOLO
from deep_sentinel.utils import logging_utils
from deep_sentinel.ai.training.augmentor import DataAugmentor
from deep_sentinel.ai.training.validator import ModelValidator

logger = logging_utils.setup_module_logger(__name__)

class ModelTrainer:
    """Handles end-to-end model training for threat detection
    
    Attributes:
        config: Training configuration dictionary
        data_path: Path to training dataset
        output_dir: Directory to save trained models
        model: YOLO model instance
        augmentor: Data augmentation module
        validator: Model validation module
    """
    
    def __init__(self, config, data_path, output_dir='models/custom'):
        """
        Initialize model trainer
        
        Args:
            config: Training configuration from model_config.yaml
            data_path: Path to dataset directory
            output_dir: Output directory for trained models
        """
        self.config = config
        self.data_path = data_path
        self.output_dir = output_dir
        self.augmentor = DataAugmentor(config)
        self.validator = ModelValidator(config)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize model
        self.model = YOLO(config['model']['base'])
        logger.info(f"Initialized trainer with base model: {config['model']['base']}")
    
    def train(self, epochs=100, batch_size=16, validation_split=0.2):
        """
        Train the threat detection model
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction of data for validation
        """
        # Prepare dataset configuration
        self._prepare_dataset_config(validation_split)
        
        # Set training parameters
        train_args = {
            'data': os.path.join(self.data_path, 'dataset.yaml'),
            'epochs': epochs,
            'batch': batch_size,
            'imgsz': self.config['model']['input_size'],
            'patience': self.config['transfer'].get('patience', 50),
            'save': True,
            'exist_ok': True,
            'project': self.output_dir,
            'name': 'train',
            'optimizer': 'AdamW',
            'lr0': self.config['training']['learning_rate'],
            'weight_decay': self.config['training']['weight_decay'],
            'momentum': self.config['training']['momentum'],
            'warmup_epochs': self.config['training']['warmup_epochs'],
            'warmup_momentum': self.config['training']['warmup_momentum'],
            'warmup_bias_lr': self.config['training']['warmup_bias_lr'],
            'box': self.config['training']['box_loss_gain'],
            'cls': self.config['training']['cls_loss_gain'],
            'obj': self.config['training']['obj_loss_gain'],
            'iou': self.config['training']['iou_threshold'],
            'anchor_t': self.config['training']['anchor_t'],
            'hsv_h': self.config['augmentation']['hsv_h'],
            'hsv_s': self.config['augmentation']['hsv_s'],
            'hsv_v': self.config['augmentation']['hsv_v'],
            'translate': self.config['augmentation']['translate'],
            'scale': self.config['augmentation']['scale'],
            'fliplr': self.config['augmentation']['fliplr'],
            'mosaic': self.config['augmentation']['mosaic'],
            'mixup': self.config['augmentation']['mixup'],
        }
        
        # Freeze layers if specified
        freeze = self.config['transfer'].get('freeze', 0)
        if freeze > 0:
            self.model.freeze(freeze)
            logger.info(f"Froze first {freeze} layers")
        
        # Start training
        start_time = time.time()
        try:
            results = self.model.train(**train_args)
            logger.info(f"Training completed in {time.time() - start_time:.2f} seconds")
            
            # Validate the model
            metrics = self.validator.validate(
                self.model, 
                os.path.join(self.data_path, 'dataset.yaml')
            )
            return metrics
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise
    
    def save_model(self, format='onnx'):
        """
        Save trained model
        
        Args:
            format: Export format ('onnx', 'torchscript', 'pt')
        """
        model_path = os.path.join(self.output_dir, 'train', 'weights', 'best.pt')
        if not os.path.exists(model_path):
            logger.error("No trained model found to export")
            return False
        
        # Load best model
        model = YOLO(model_path)
        
        # Export model
        export_path = os.path.join(self.output_dir, f"deep_sentinel.{format}")
        success = model.export(
            format=format,
            imgsz=self.config['model']['input_size'],
            half=self.config['save'].get('half_precision', True),
            int8=self.config['save'].get('quantize', False),
            simplify=True,
            opset=12,
            dynamic=False,
            name=export_path
        )
        
        if success:
            logger.info(f"Exported model to {export_path}")
            return True
        else:
            logger.error("Model export failed")
            return False
    
    def _prepare_dataset_config(self, validation_split):
        """Create dataset YAML file for YOLO training"""
        dataset_config = {
            'path': self.data_path,
            'train': self.config['dataset']['train'],
            'val': self.config['dataset']['val'],
            'names': self.config['dataset']['names']
        }
        
        # Save dataset config
        with open(os.path.join(self.data_path, 'dataset.yaml'), 'w') as f:
            yaml.dump(dataset_config, f)
        
        logger.info("Prepared dataset configuration")