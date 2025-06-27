import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from deep_sentinel.utils import logging_utils

logger = logging_utils.setup_module_logger(__name__)

class ModelValidator:
    """Validates trained threat detection models
    
    Attributes:
        config: Validation configuration dictionary
    """
    
    def __init__(self, config):
        """
        Initialize model validator
        
        Args:
            config: Validation configuration from model_config.yaml
        """
        self.config = config['validation']
        logger.info("Model validator initialized")
    
    def validate(self, model, data_yaml):
        """
        Validate model performance
        
        Args:
            model: Trained YOLO model
            data_yaml: Path to dataset YAML file
            
        Returns:
            dict: Validation metrics
        """
        # Run validation
        metrics = model.val(
            data=data_yaml,
            imgsz=self.config['imgsz'],
            batch=self.config['batch_size'],
            conf=self.config['conf_threshold'],
            iou=self.config['iou_threshold'],
            max_det=self.config['max_det'],
            plots=self.config['plots'],
            save_json=True,
            save_hybrid=True,
            half=True
        )
        
        # Save detailed report
        self._save_validation_report(metrics)
        return metrics
    
    def _save_validation_report(self, metrics):
        """Generate and save comprehensive validation report"""
        report_dir = os.path.join(metrics.save_dir, 'validation_report')
        os.makedirs(report_dir, exist_ok=True)
        
        # 1. Save metrics summary
        with open(os.path.join(report_dir, 'summary.txt'), 'w') as f:
            f.write(f"Validation Report\n")
            f.write(f"----------------\n\n")
            f.write(f"Precision: {metrics.results_dict['metrics/precision(B)']:.4f}\n")
            f.write(f"Recall: {metrics.results_dict['metrics/recall(B)']:.4f}\n")
            f.write(f"mAP@0.5: {metrics.results_dict['metrics/mAP50(B)']:.4f}\n")
            f.write(f"mAP@0.5-0.95: {metrics.results_dict['metrics/mAP50-95(B)']:.4f}\n\n")
            
            f.write("Per-Class Metrics:\n")
            for i, name in enumerate(metrics.names.values()):
                f.write(f"{name}:\n")
                f.write(f"  Precision: {metrics.class_result(i)['precision']:.4f}\n")
                f.write(f"  Recall: {metrics.class_result(i)['recall']:.4f}\n")
                f.write(f"  mAP@0.5: {metrics.class_result(i)['map50']:.4f}\n")
                f.write(f"  mAP@0.5-0.95: {metrics.class_result(i)['map']:.4f}\n")
        
        # 2. Generate confusion matrix
        plt.figure(figsize=(12, 10))
        metrics.confusion_matrix.plot(normalize=True, save_dir=report_dir)
        plt.close()
        
        # 3. Generate F1 curve
        plt.figure()
        metrics.f1_curve.plot(save_dir=report_dir)
        plt.close()
        
        # 4. Generate PR curve
        plt.figure()
        metrics.precision_curve.plot(save_dir=report_dir)
        plt.close()
        
        # 5. Generate ROC curve
        plt.figure()
        metrics.roc_curve.plot(save_dir=report_dir)
        plt.close()
        
        # 6. Generate detection examples
        self._plot_detection_examples(metrics, report_dir)
        
        logger.info(f"Validation report saved to {report_dir}")
    
    def _plot_detection_examples(self, metrics, report_dir):
        """Plot example detections from validation set"""
        examples_dir = os.path.join(report_dir, 'examples')
        os.makedirs(examples_dir, exist_ok=True)
        
        # Get sample validation images
        val_images_dir = os.path.join(metrics.data['path'], metrics.data['val'])
        image_files = [f for f in os.listdir(val_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Process sample images
        for i, img_file in enumerate(image_files[:10]):  # First 10 images
            img_path = os.path.join(val_images_dir, img_file)
            results = metrics.model(img_path)
            
            # Plot and save
            for r in results:
                r.save(filename=os.path.join(examples_dir, f"result_{i}.jpg"))
    
    def compare_models(self, model_paths, data_yaml):
        """
        Compare multiple models and generate report
        
        Args:
            model_paths: List of paths to model files
            data_yaml: Path to dataset YAML file
            
        Returns:
            dict: Comparison results
        """
        comparison = {}
        
        for model_path in model_paths:
            model = YOLO(model_path)
            metrics = model.val(
                data=data_yaml,
                imgsz=self.config['imgsz'],
                conf=self.config['conf_threshold'],
                iou=self.config['iou_threshold'],
                plots=False
            )
            
            comparison[model_path] = {
                'mAP50': metrics.results_dict['metrics/mAP50(B)'],
                'mAP50-95': metrics.results_dict['metrics/mAP50-95(B)'],
                'precision': metrics.results_dict['metrics/precision(B)'],
                'recall': metrics.results_dict['metrics/recall(B)']
            }
        
        # Generate comparison plot
        self._plot_comparison(comparison)
        return comparison
    
    def _plot_comparison(self, comparison):
        """Plot model comparison results"""
        models = list(comparison.keys())
        map50 = [comp['mAP50'] for comp in comparison.values()]
        map5095 = [comp['mAP50-95'] for comp in comparison.values()]
        
        x = np.arange(len(models))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 8))
        rects1 = ax.bar(x - width/2, map50, width, label='mAP@0.5')
        rects2 = ax.bar(x + width/2, map5095, width, label='mAP@0.5-0.95')
        
        ax.set_ylabel('mAP Score')
        ax.set_title('Model Comparison by mAP')
        ax.set_xticks(x)
        ax.set_xticklabels([os.path.basename(m) for m in models], rotation=45)
        ax.legend()
        
        fig.tight_layout()
        plt.savefig('model_comparison.png', bbox_inches='tight')
        plt.close()