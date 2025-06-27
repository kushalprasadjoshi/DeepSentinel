import os
import json
import logging
from deep_sentinel.utils import logging_utils

logger = logging_utils.setup_module_logger(__name__)

class ModelUpdater:
    """Manages model updates from cloud storage
    
    Attributes:
        aws_client: AWSClient instance
        model_dir: Local directory for models
        current_version: Current model version
    """
    
    def __init__(self, aws_client, model_dir='models'):
        self.aws_client = aws_client
        self.model_dir = model_dir
        self.current_version = self._load_current_version()
        logger.info("Model updater initialized")
    
    def _load_current_version(self):
        """Load current model version from version file"""
        version_file = os.path.join(self.model_dir, 'version.json')
        if os.path.exists(version_file):
            try:
                with open(version_file, 'r') as f:
                    data = json.load(f)
                return data.get('version', '1.0.0')
            except Exception:
                pass
        return '1.0.0'
    
    def check_for_updates(self, s3_bucket, s3_prefix):
        """
        Check for new model versions
        
        Args:
            s3_bucket: S3 bucket containing models
            s3_prefix: S3 prefix for model files
            
        Returns:
            list: Available versions newer than current
        """
        try:
            # List model versions in S3
            response = self.aws_client.s3.list_objects_v2(
                Bucket=s3_bucket,
                Prefix=s3_prefix
            )
            
            # Extract versions
            versions = []
            for obj in response.get('Contents', []):
                if obj['Key'].endswith('.pt'):
                    version = obj['Key'].split('/')[-1].replace('.pt', '')
                    if version > self.current_version:
                        versions.append(version)
            
            return sorted(versions, reverse=True)
        except Exception as e:
            logger.error(f"Update check failed: {str(e)}")
            return []
    
    def download_model(self, s3_bucket, s3_key, version):
        """
        Download model from S3
        
        Args:
            s3_bucket: S3 bucket name
            s3_key: S3 object key
            version: Model version
            
        Returns:
            bool: True if successful
        """
        local_path = os.path.join(self.model_dir, f"model_{version}.pt")
        try:
            self.aws_client.s3.download_file(s3_bucket, s3_key, local_path)
            
            # Update version file
            with open(os.path.join(self.model_dir, 'version.json'), 'w') as f:
                json.dump({'version': version}, f)
            
            self.current_version = version
            logger.info(f"Updated to model version {version}")
            return True
        except Exception as e:
            logger.error(f"Model download failed: {str(e)}")
            return False
    
    def update_model(self, s3_bucket, s3_prefix):
        """
        Update to the latest model version
        
        Args:
            s3_bucket: S3 bucket name
            s3_prefix: S3 prefix for models
            
        Returns:
            bool: True if updated
        """
        available_versions = self.check_for_updates(s3_bucket, s3_prefix)
        if not available_versions:
            logger.info("No updates available")
            return False
        
        latest_version = available_versions[0]
        s3_key = f"{s3_prefix}/{latest_version}.pt"
        return self.download_model(s3_bucket, s3_key, latest_version)