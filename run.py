import sys
import signal
from deep_sentinel.interfaces.gui.main_window import MainApplication
from deep_sentinel.utils.config_loader import load_config
from deep_sentinel.utils.logging_utils import setup_logger

def main():
    """Main application entry point"""
    # Setup logging
    logger = setup_logger("DeepSentinel")
    logger.info("Starting DeepSentinel Security System")
    
    try:
        # Load configuration
        config = load_config("app")
        if not config:
            logger.error("Failed to load configuration. Exiting.")
            return
        
        # Create and run application
        app = MainApplication(config)
        app.mainloop()
        
    except Exception as e:
        logger.exception(f"Critical error: {str(e)}")
        sys.exit(1)
    
    logger.info("Application exited")

def signal_handler(sig, frame):
    """Handle shutdown signals"""
    print("\nShutting down...")
    # Additional cleanup would go here
    sys.exit(0)

if __name__ == "__main__":
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # System shutdown
    
    # Start application
    main()