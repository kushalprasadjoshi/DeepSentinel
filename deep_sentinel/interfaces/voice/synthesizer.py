import pyttsx3
import queue
import threading
from deep_sentinel.utils import logging_utils

logger = logging_utils.setup_module_logger(__name__)

class VoiceSynthesizer:
    """Converts text to speech using pyttsx3
    
    Attributes:
        engine: TTS engine
        message_queue: Queue for messages to speak
        running: Flag indicating if synthesizer is active
        thread: Processing thread
    """
    
    def __init__(self, rate=150, volume=1.0, voice=None):
        self.engine = pyttsx3.init()
        self.set_rate(rate)
        self.set_volume(volume)
        if voice:
            self.set_voice(voice)
        self.message_queue = queue.Queue()
        self.running = False
        self.thread = None
        logger.info("Voice synthesizer initialized")
    
    def set_rate(self, rate):
        """Set speech rate (words per minute)"""
        self.engine.setProperty('rate', rate)
    
    def set_volume(self, volume):
        """Set volume level (0.0 to 1.0)"""
        self.engine.setProperty('volume', volume)
    
    def set_voice(self, voice_id):
        """Set voice by ID"""
        voices = self.engine.getProperty('voices')
        if voice_id < len(voices):
            self.engine.setProperty('voice', voices[voice_id].id)
    
    def speak(self, text):
        """Add text to speak queue"""
        self.message_queue.put(text)
        logger.debug(f"Added to speech queue: {text}")
    
    def start(self):
        """Start processing speech queue"""
        if self.running:
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._process_queue)
        self.thread.daemon = True
        self.thread.start()
        logger.info("Speech synthesis started")
    
    def stop(self):
        """Stop processing speech queue"""
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        logger.info("Speech synthesis stopped")
    
    def _process_queue(self):
        """Process messages from the queue"""
        while self.running:
            try:
                text = self.message_queue.get(timeout=1.0)
                self.engine.say(text)
                self.engine.runAndWait()
            except queue.Empty:
                pass
            except Exception as e:
                logger.error(f"Speech synthesis error: {str(e)}")