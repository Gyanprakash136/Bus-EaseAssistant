"""
TTS Engine - Python 3.13 Compatible
"""
import asyncio
import os
import uuid
from typing import Dict, Any, Optional
import logging
from config.settings import settings

try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False

logger = logging.getLogger(__name__)

class TTSEngine:
    """Text-to-Speech engine."""
    
    def __init__(self):
        self.engine = getattr(settings, 'tts_engine', 'edge_tts')
        self.temp_dir = getattr(settings, 'audio_temp_dir', 'temp/audio')
        
        # Voice settings
        self.voices = {
            'en': getattr(settings, 'tts_voice_en', 'en-US-AriaNeural'),
            'hi': getattr(settings, 'tts_voice_hi', 'hi-IN-SwaraNeural'),
            'pa': getattr(settings, 'tts_voice_pa', 'pa-IN-GulNeural')
        }
        
        os.makedirs(self.temp_dir, exist_ok=True)
        
        if not EDGE_TTS_AVAILABLE:
            self.engine = 'mock'
            logger.warning("Edge TTS not available - using mock TTS")
        
        logger.info(f"TTSEngine initialized - Engine: {self.engine}")
    
    async def text_to_speech(self, text: str, language: str = 'en') -> Dict[str, Any]:
        """Convert text to speech."""
        try:
            if self.engine == 'edge_tts' and EDGE_TTS_AVAILABLE:
                return await self.edge_tts_synthesis(text, language)
            else:
                return self.mock_tts(text, language)
        except Exception as e:
            logger.error(f"TTS error: {e}")
            return {
                "success": False,
                "error": str(e),
                "audio_file": None,
                "audio_url": None
            }
    
    async def edge_tts_synthesis(self, text: str, language: str) -> Dict[str, Any]:
        """Generate speech using Edge TTS."""
        try:
            voice = self.voices.get(language, self.voices['en'])
            filename = f"tts_{uuid.uuid4().hex[:8]}.wav"
            filepath = os.path.join(self.temp_dir, filename)
            
            # Generate speech
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(filepath)
            
            # Create URL for serving
            audio_url = f"/static/audio/{filename}"
            
            # Copy to static directory for serving
            static_audio_dir = "static/audio"
            os.makedirs(static_audio_dir, exist_ok=True)
            static_filepath = os.path.join(static_audio_dir, filename)
            
            # Copy file
            with open(filepath, 'rb') as src, open(static_filepath, 'wb') as dst:
                dst.write(src.read())
            
            return {
                "success": True,
                "text": text,
                "language": language,
                "voice": voice,
                "audio_file": static_filepath,
                "audio_url": audio_url,
                "engine": "edge_tts"
            }
            
        except Exception as e:
            logger.error(f"Edge TTS error: {e}")
            return self.mock_tts(text, language)
    
    def mock_tts(self, text: str, language: str) -> Dict[str, Any]:
        """Mock TTS for testing."""
        filename = f"mock_tts_{uuid.uuid4().hex[:8]}.wav"
        audio_url = f"/static/audio/{filename}"
        
        return {
            "success": True,
            "text": text,
            "language": language,
            "voice": "mock",
            "audio_file": None,
            "audio_url": audio_url,
            "engine": "mock",
            "message": "Mock TTS - no actual audio generated"
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get TTS status."""
        return {
            "engine": self.engine,
            "edge_tts_available": EDGE_TTS_AVAILABLE,
            "supported_languages": list(self.voices.keys()),
            "voices": self.voices
        }

# Global instance
tts_engine = TTSEngine()
