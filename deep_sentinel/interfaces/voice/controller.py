import speech_recognition as sr
import threading
import time
import re
import hashlib
from deep_sentinel.utils import logging_utils

logger = logging_utils.setup_module_logger(__name__)

class VoiceController:
    """Handles voice command recognition and processing
    
    Attributes:
        recognizer: Speech recognizer instance
        microphone: Microphone instance
        commands: Dictionary of command patterns and handlers
        wake_word: Wake word to activate listening
        is_listening: Flag indicating if listening is active
        auth_users: Dictionary of authorized users with voiceprints
    """
    
    def __init__(self, wake_word="sentinel", auth_users=None):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.wake_word = wake_word.lower()
        self.commands = {}
        self.is_listening = False
        self.auth_users = auth_users or {}
        self.listening_thread = None
        self.active_user = None
        
        # Adjust for ambient noise
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
        
        logger.info("Voice controller initialized")
    
    def register_command(self, pattern, handler, requires_auth=True):
        """
        Register a voice command pattern and handler
        
        Args:
            pattern: Regular expression pattern to match command
            handler: Function to call when command is recognized
            requires_auth: Whether the command requires authentication
        """
        self.commands[re.compile(pattern, re.IGNORECASE)] = {
            'handler': handler,
            'requires_auth': requires_auth
        }
        logger.debug(f"Registered command: {pattern}")
    
    def start_listening(self):
        """Start continuous voice command listening"""
        if self.is_listening:
            return
            
        self.is_listening = True
        self.listening_thread = threading.Thread(target=self._listen_loop)
        self.listening_thread.daemon = True
        self.listening_thread.start()
        logger.info("Voice listening started")
    
    def stop_listening(self):
        """Stop voice command listening"""
        self.is_listening = False
        if self.listening_thread is not None:
            self.listening_thread.join(timeout=2.0)
        logger.info("Voice listening stopped")
    
    def _listen_loop(self):
        """Main listening loop"""
        while self.is_listening:
            try:
                # Listen for audio
                with self.microphone as source:
                    audio = self.recognizer.listen(source, timeout=3, phrase_time_limit=2)
                
                try:
                    # Recognize speech
                    text = self.recognizer.recognize_google(audio).lower()
                    logger.debug(f"Recognized: {text}")
                    
                    # Check for wake word
                    if self.wake_word in text:
                        logger.info("Wake word detected")
                        self.active_user = None  # Reset authentication
                        self.process_command()
                        
                except sr.UnknownValueError:
                    logger.debug("Could not understand audio")
                except sr.RequestError as e:
                    logger.error(f"Recognition error: {str(e)}")
                    
            except sr.WaitTimeoutError:
                # Timeout, just continue
                pass
            
            # Short delay to prevent CPU overload
            time.sleep(0.1)
    
    def process_command(self):
        """Listen for and process a command after wake word"""
        with self.microphone as source:
            logger.debug("Listening for command...")
            audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=3)
        
        try:
            command_text = self.recognizer.recognize_google(audio).lower()
            logger.info(f"Command received: {command_text}")
            
            # Check authentication requirements
            requires_auth = False
            matched_handler = None
            
            # Find matching command
            for pattern, command_data in self.commands.items():
                if pattern.match(command_text):
                    matched_handler = command_data['handler']
                    requires_auth = command_data['requires_auth']
                    break
            
            if matched_handler:
                # Handle authentication if required
                if requires_auth and not self.active_user:
                    logger.info("Command requires authentication")
                    if self.authenticate_user():
                        matched_handler(command_text)
                else:
                    matched_handler(command_text)
            else:
                logger.warning(f"No handler for command: {command_text}")
            
        except sr.UnknownValueError:
            logger.debug("Could not understand command")
        except sr.RequestError as e:
            logger.error(f"Recognition error: {str(e)}")
    
    def authenticate_user(self):
        """Authenticate user via voice recognition"""
        logger.info("Please say your authentication phrase")
        
        with self.microphone as source:
            audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=3)
        
        try:
            # Get voice sample
            audio_data = audio.get_wav_data()
            voice_hash = hashlib.sha256(audio_data).hexdigest()
            
            # Compare with registered users
            for user_id, user_data in self.auth_users.items():
                if user_data['voice_hash'] == voice_hash:
                    self.active_user = user_id
                    logger.info(f"Authenticated as {user_data['name']}")
                    return True
            
            logger.warning("Authentication failed")
            return False
            
        except Exception as e:
            logger.error(f"Authentication error: {str(e)}")
            return False
    
    def register_user(self, user_id, name, voice_sample):
        """
        Register a new user with voice authentication
        
        Args:
            user_id: Unique user identifier
            name: User display name
            voice_sample: Audio sample for voiceprint
        """
        audio_data = voice_sample.get_wav_data()
        voice_hash = hashlib.sha256(audio_data).hexdigest()
        
        self.auth_users[user_id] = {
            'name': name,
            'voice_hash': voice_hash
        }
        logger.info(f"Registered user: {name}")