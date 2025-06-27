import boto3
import logging
from botocore.exceptions import ClientError
from deep_sentinel.utils import logging_utils

logger = logging_utils.setup_module_logger(__name__)

class AWSClient:
    """Provides access to AWS services
    
    Attributes:
        s3: S3 client
        rekognition: Rekognition client
        sns: SNS client
    """
    
    def __init__(self, access_key, secret_key, region='us-east-1'):
        self.session = boto3.Session(
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region
        )
        self.s3 = self.session.client('s3')
        self.rekognition = self.session.client('rekognition')
        self.sns = self.session.client('sns')
        logger.info("AWS client initialized")
    
    def upload_to_s3(self, file, bucket, key):
        """
        Upload file to S3
        
        Args:
            file: File path or bytes
            bucket: S3 bucket name
            key: S3 object key
            
        Returns:
            bool: True if successful
        """
        try:
            if isinstance(file, str):
                # File path
                self.s3.upload_file(file, bucket, key)
            else:
                # File bytes
                self.s3.put_object(Body=file, Bucket=bucket, Key=key)
            logger.info(f"Uploaded to s3://{bucket}/{key}")
            return True
        except ClientError as e:
            logger.error(f"S3 upload failed: {str(e)}")
            return False
    
    def detect_objects(self, image, bucket=None, key=None):
        """
        Detect objects in image using Rekognition
        
        Args:
            image: Image bytes or S3 reference
            bucket: S3 bucket name (if using S3 reference)
            key: S3 object key (if using S3 reference)
            
        Returns:
            list: Detected objects
        """
        if bucket and key:
            # Use S3 reference
            response = self.rekognition.detect_labels(
                Image={'S3Object': {'Bucket': bucket, 'Name': key}}
            )
        else:
            # Use image bytes
            response = self.rekognition.detect_labels(
                Image={'Bytes': image}
            )
        
        return response.get('Labels', [])
    
    def send_sns_notification(self, topic_arn, message):
        """
        Send SNS notification
        
        Args:
            topic_arn: SNS topic ARN
            message: Message text
            
        Returns:
            bool: True if successful
        """
        try:
            self.sns.publish(
                TopicArn=topic_arn,
                Message=message
            )
            logger.info(f"SNS notification sent to {topic_arn}")
            return True
        except ClientError as e:
            logger.error(f"SNS send failed: {str(e)}")
            return False