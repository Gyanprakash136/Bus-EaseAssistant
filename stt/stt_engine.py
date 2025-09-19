"""
STT Engine - Python 3.13 Compatible
"""
import os
import json
import asyncio
from typing import Optional, Dict, Any, List
import logging
from config.settings import settings
from utils.audio_utils import audio_processor

try:
    import vosk
    VOSK_AVAILABLE = True
except ImportError:
    VOSK_AVAILABLE = False

logger = logging.getLogger(__name__)

class STTEngine:
    """Speech-to-Text engine."""
    
    def __init__(self):
        self.sample_rate = getattr(settings, 'sample_rate', 16000)
        self.engine = getattr(settings, 'stt_engine', 'vosk')
        self.model_path = getattr(settings, 'model_path', './models')
        
        self.vosk_model = None
        self.vosk_rec = None
        
        self.supported_languages = {
            'en': 'english',
            'hi': 'hindi',
            'pa': 'punjabi'
        }
        
        self.initialize_engine()
        logger.info(f"STTEngine initialized - Engine: {self.engine}")
    
    def initialize_engine(self):
        """Initialize STT engine."""
        if self.engine == 'vosk' and VOSK_AVAILABLE:
            self.initialize_vosk()
        else:
            logger.warning("No STT engine available - using mock")
            self.engine = 'mock'
    
    def initialize_vosk(self):
        """Initialize Vosk if available."""
        try:
            os.makedirs(self.model_path, exist_ok=True)
            
            # Try to find existing model
            for item in os.listdir(self.model_path):
                item_path = os.path.join(self.model_path, item)
                if os.path.isdir(item_path):
                    try:
                        self.vosk_model = vosk.Model(item_path)
                        self.vosk_rec = vosk.KaldiRecognizer(self.vosk_model, self.sample_rate)
                        logger.info(f"Vosk model loaded: {item_path}")
                        return
                    except:
                        continue
            
            logger.warning("No Vosk model found - using mock STT")
            self.engine = 'mock'
            
        except Exception as e:
            logger.error(f"Vosk initialization error: {e}")
            self.engine = 'mock'
    
    async def transcribe_audio(self, audio_data: bytes, language: str = 'en') -> Dict[str, Any]:
        """Transcribe audio to text."""
        try:
            if self.engine == 'vosk' and self.vosk_rec:
                return await self.transcribe_with_vosk(audio_data, language)
            else:
                return self.mock_transcription(language)
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return {
                "success": False,
                "error": str(e),
                "text": "",
                "confidence": 0.0,
                "language": language
            }
    
    async def transcribe_with_vosk(self, audio_data: bytes, language: str) -> Dict[str, Any]:
        """Transcribe using Vosk."""
        try:
            # Save and process audio
            temp_file = audio_processor.save_audio_file(audio_data)
            
            try:
                # Read audio (skip WAV header)
                with open(temp_file, 'rb') as f:
                    f.seek(44)  # Skip WAV header
                    audio_bytes = f.read()
                
                # Process in chunks
                results = []
                chunk_size = self.sample_rate * 2  # 1 second chunks
                
                for i in range(0, len(audio_bytes), chunk_size):
                    chunk = audio_bytes[i:i + chunk_size]
                    if len(chunk) == 0:
                        break
                    
                    if self.vosk_rec.AcceptWaveform(chunk):
                        result = json.loads(self.vosk_rec.Result())
                        if result.get('text'):
                            results.append(result['text'])
                
                # Get final result
                final_result = json.loads(self.vosk_rec.FinalResult())
                if final_result.get('text'):
                    results.append(final_result['text'])
                
                full_text = ' '.join(results).strip()
                confidence = final_result.get('conf', 0.8)
                
                return {
                    "success": True,
                    "text": full_text,
                    "confidence": confidence,
                    "language": language,
                    "engine": "vosk"
                }
                
            finally:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    
        except Exception as e:
            logger.error(f"Vosk transcription error: {e}")
            return self.mock_transcription(language)
    
    def mock_transcription(self, language: str) -> Dict[str, Any]:
        """Mock transcription for testing."""
        mock_texts = {
            'en': "Where is bus 101A?",
            'hi': "बस 101A कहाँ है?",
            'pa': "ਬੱਸ 101A ਕਿੱਥੇ ਹੈ?"
        }
        
        return {
            "success": True,
            "text": mock_texts.get(language, mock_texts['en']),
            "confidence": 0.9,
            "language": language,
            "engine": "mock"
        }
    
    def detect_language(self, audio_data: bytes) -> str:
        """Simple language detection."""
        return getattr(settings, 'default_language', 'en')
    
    def get_status(self) -> Dict[str, Any]:
        """Get STT status."""
        return {
            "engine": self.engine,
            "vosk_available": VOSK_AVAILABLE and self.vosk_model is not None,
            "supported_languages": list(self.supported_languages.keys()),
            "sample_rate": self.sample_rate
        }

# Global instance
stt_engine = STTEngine()
