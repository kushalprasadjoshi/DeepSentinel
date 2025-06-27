import argparse
import yaml
from deep_sentinel.ai.training.trainer import ModelTrainer
from deep_sentinel.utils.logging_utils import setup_logger

def main():
    """Model training entry point"""
    # Setup logger
    logger = setup_logger("Training")
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train DeepSentinel threat detection model")
    parser.add_argument('--config', default='config/model_config.yaml', help='Model configuration file')
    parser.add_argument('--data', default='data/processed', help='Path to training data')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--output', default='models/custom', help='Output directory for trained model')
    args = parser.parse_args()
    
    try:
        # Load model configuration
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Starting training with config: {args.config}")
        logger.info(f"Training data: {args.data}")
        logger.info(f"Epochs: {args.epochs}, Batch size: {args.batch}")
        
        # Initialize trainer
        trainer = ModelTrainer(config, args.data, args.output)
        
        # Run training
        trainer.train(
            epochs=args.epochs,
            batch_size=args.batch,
            validation_split=0.2
        )
        
        # Save final model
        trainer.save_model()
        logger.info("Training completed successfully")
        
    except Exception as e:
        logger.exception(f"Training failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()