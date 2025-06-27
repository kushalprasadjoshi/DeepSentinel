import logging
from logging.handlers import RotatingFileHandler
import os

def setup_module_logger(module_name, log_file="logs/deepsentinel.log", level=logging.INFO):
    """
    Set up a logger for a specific module with file rotation
    
    Args:
        module_name: Name of the module (e.g., __name__)
        log_file: Path to log file
        level: Logging level
        
    Returns:
        Logger: Configured logger instance
    """
    # Create logs directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Create logger with module name
    logger = logging.getLogger(module_name)
    logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create rotating file handler (10MB per file, keep 5 backups)
    file_handler = RotatingFileHandler(
        log_file, maxBytes=10*1024*1024, backupCount=5
    )
    file_handler.setFormatter(formatter)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger