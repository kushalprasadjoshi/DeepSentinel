import yaml
import os
from deep_sentinel import logger

# Configuration cache
_config_cache = {}

def load_config(config_name="app", config_dir="../config"):
    """
    Load configuration from YAML file with environment variable override
    
    Args:
        config_name: Base name of config file (without .yaml extension)
        config_dir: Directory containing config files
        
    Returns:
        dict: Loaded configuration dictionary
    """
    global _config_cache
    
    # Check cache first
    if config_name in _config_cache:
        return _config_cache[config_name]
    
    config_path = os.path.join(config_dir, f"{config_name}_config.yaml")
    
    try:
        # Load configuration from YAML file
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Apply environment variable overrides
        for key, value in os.environ.items():
            if key.startswith("DS_"):
                # Convert DS_SECTION_KEY to [section][key]
                parts = key[3:].lower().split('_')
                if len(parts) >= 2:
                    section = parts[0]
                    subkey = '_'.join(parts[1:])
                    if section in config and subkey in config[section]:
                        config[section][subkey] = value
                        logger.info(f"Overridden {section}.{subkey} from environment")
        
        # Cache the configuration
        _config_cache[config_name] = config
        logger.info(f"Loaded configuration from {config_path}")
        return config
    
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        return {}
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML config: {str(e)}")
        return {}
    except Exception as e:
        logger.error(f"Unexpected error loading config: {str(e)}")
        return {}