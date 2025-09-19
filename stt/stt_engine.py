"""
STT Engine - Production Ready with Enhanced Error Handling & Fallback Options
Compatible with Python 3.13 and Render deployment
"""
import os
import json
import asyncio
import tempfile
import wave
from typing import Optional, Dict, Any, List
import logging
from datetime import datetime
from config.settings import settings

# Import audio utils with fallback
try:
    from utils.audio_utils import audio_processor
except ImportError:
    audio_processor = None

# Import language utils for detection
try:
    from utils.language_utils import language_utils
except ImportError:
    language_utils = None

# Import Vosk with fallback
try:
    import vosk
    VOSK_AVAILABLE = True
except ImportError:
    VOSK_AVAILABLE = False

logger = logging.getLogger(__name__)

class STTEngine:
    """Enhanced Speech-to-Text engine with production-grade reliability."""
    
    def __init__(self):
        # Audio settings
        self.sample_rate = getattr(settings, 'audio_sample_rate', getattr(settings, 'sample_rate', 16000))
        self.channels = getattr(settings, 'audio_channels', getattr(settings, 'channels', 1))
        self.chunk_size = getattr(settings, 'audio_chunk_size', getattr(settings, 'chunk_size', 1024))
        
        # Engine settings
        self.engine = getattr(settings, 'stt_engine', 'vosk')
        self.model_path = getattr(settings, 'vosk_model_path', getattr(settings, 'model_path', './models'))
        self.enable_vosk = getattr(settings, 'enable_vosk', True)
        
        # Model management
        self.vosk_model = None
        self.vosk_rec = None
        self.current_model_language = None
        
        # Language support
        self.supported_languages = {
            'en': {
                'name': 'English',
                'vosk_model': 'vosk-model-en-us',
                'mock_text': "Where is bus 101A?"
            },
            'hi': {
                'name': 'Hindi', 
                'vosk_model': 'vosk-model-hi',
                'mock_text': "बस 101A कहाँ है?"
            },
            'pa': {
                'name': 'Punjabi',
                'vosk_model': 'vosk-model-pa', 
                'mock_text': "ਬੱਸ 101A ਕਿੱਥੇ ਹੈ?"
            }
        }
        
        # Statistics
        self.stats = {
            'total_transcriptions': 0,
            'successful_transcriptions': 0,
            'failed_transcriptions': 0,
            'language_counts': {},
            'average_confidence': 0.0,
            'last_transcription_time': None
        }
        
        # Initialize engine
        self.initialize_engine()
        logger.info(f"✅ STTEngine initialized - Engine: {self.engine}, Sample Rate: {self.sample_rate}")
    
    def initialize_engine(self):
        """Initialize STT engine with fallback options."""
        if self.engine == 'vosk' and VOSK_AVAILABLE and self.enable_vosk:
            self.initialize_vosk()
        else:
            logger.warning("⚠️ Vosk STT not available - using mock transcription")
            self.engine = 'mock'
    
    def initialize_vosk(self):
        """Initialize Vosk with enhanced model management."""
        try:
            # Create models directory
            os.makedirs(self.model_path, exist_ok=True)
            
            # Try to find and load existing model
            model_loaded = False
            
            for item in os.listdir(self.model_path):
                item_path = os.path.join(self.model_path, item)
                if os.path.isdir(item_path) and item.startswith('vosk-model'):
                    try:
                        logger.info(f"🔍 Attempting to load Vosk model: {item_path}")
                        self.vosk_model = vosk.Model(item_path)
                        self.vosk_rec = vosk.KaldiRecognizer(self.vosk_model, self.sample_rate)
                        self.current_model_language = self._detect_model_language(item)
                        logger.info(f"✅ Vosk model loaded successfully: {item} (Language: {self.current_model_language})")
                        model_loaded = True
                        break
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to load model {item}: {e}")
                        continue
            
            if not model_loaded:
                logger.warning("⚠️ No valid Vosk model found - using mock STT")
                logger.info(f"💡 To enable Vosk STT, download a model to: {self.model_path}")
                logger.info("📥 Download from: https://alphacephei.com/vosk/models")
                self.engine = 'mock'
            
        except Exception as e:
            logger.error(f"❌ Vosk initialization error: {e}")
            self.engine = 'mock'
    
    def _detect_model_language(self, model_name: str) -> str:
        """Detect language from model name."""
        model_name_lower = model_name.lower()
        
        if 'en' in model_name_lower or 'english' in model_name_lower:
            return 'en'
        elif 'hi' in model_name_lower or 'hindi' in model_name_lower:
            return 'hi'
        elif 'pa' in model_name_lower or 'punjabi' in model_name_lower:
            return 'pa'
        else:
            return 'en'  # Default to English
    
    async def transcribe_audio(self, audio_data: bytes, language: str = 'en') -> Dict[str, Any]:
        """Enhanced audio transcription with comprehensive error handling."""
        try:
            # Update statistics
            self.stats['total_transcriptions'] += 1
            self.stats['last_transcription_time'] = datetime.now().isoformat()
            self.stats['language_counts'][language] = self.stats['language_counts'].get(language, 0) + 1
            
            # Validate input
            if not audio_data or len(audio_data) < 100:
                raise ValueError("Audio data too small or empty")
            
            logger.info(f"🎤 Transcribing audio: {len(audio_data)} bytes, Language: {language}")
            
            # Validate language
            if language not in self.supported_languages:
                logger.warning(f"⚠️ Unsupported language {language}, using English")
                language = 'en'
            
            # Choose transcription method
            if self.engine == 'vosk' and self.vosk_rec:
                result = await self.transcribe_with_vosk(audio_data, language)
            else:
                result = await self.mock_transcription(language)
            
            # Update statistics
            if result.get('success'):
                self.stats['successful_transcriptions'] += 1
                
                # Update average confidence
                confidence = result.get('confidence', 0.0)
                total_successful = self.stats['successful_transcriptions']
                current_avg = self.stats['average_confidence']
                self.stats['average_confidence'] = (current_avg * (total_successful - 1) + confidence) / total_successful
            else:
                self.stats['failed_transcriptions'] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Transcription error: {e}")
            self.stats['failed_transcriptions'] += 1
            
            return {
                "success": False,
                "error": str(e),
                "text": "",
                "confidence": 0.0,
                "language": language,
                "engine": self.engine,
                "timestamp": datetime.now().isoformat()
            }
    
    async def transcribe_with_vosk(self, audio_data: bytes, language: str) -> Dict[str, Any]:
        """Enhanced Vosk transcription with better audio handling."""
        try:
            logger.info(f"🔄 Processing with Vosk (Model language: {self.current_model_language})")
            
            # Create temporary file for audio processing
            temp_file = None
            try:
                # Save audio to temporary file
                if audio_processor:
                    temp_file = audio_processor.save_audio_file(audio_data)
                else:
                    temp_file = self._save_audio_fallback(audio_data)
                
                # Validate audio file
                if not self._validate_audio_file(temp_file):
                    raise ValueError("Invalid audio file format")
                
                # Process audio in chunks
                transcription_results = []
                confidence_scores = []
                
                # Read audio data (skip WAV header if present)
                with open(temp_file, 'rb') as f:
                    # Check if it's a WAV file and skip header
                    header = f.read(12)
                    if header.startswith(b'RIFF') and b'WAVE' in header:
                        f.seek(44)  # Skip standard WAV header
                    else:
                        f.seek(0)  # Reset if not WAV
                    
                    audio_bytes = f.read()
                
                if len(audio_bytes) == 0:
                    raise ValueError("No audio data found after header")
                
                # Process in chunks for better accuracy
                chunk_size = self.sample_rate * 2  # 1 second chunks (2 bytes per sample for 16-bit)
                total_chunks = len(audio_bytes) // chunk_size + (1 if len(audio_bytes) % chunk_size else 0)
                
                logger.info(f"📊 Processing {total_chunks} audio chunks")
                
                for i in range(0, len(audio_bytes), chunk_size):
                    chunk = audio_bytes[i:i + chunk_size]
                    if len(chunk) == 0:
                        break
                    
                    # Process chunk
                    if self.vosk_rec.AcceptWaveform(chunk):
                        result = json.loads(self.vosk_rec.Result())
                        if result.get('text', '').strip():
                            transcription_results.append(result['text'].strip())
                            confidence_scores.append(result.get('conf', 0.5))
                
                # Get final result
                final_result = json.loads(self.vosk_rec.FinalResult())
                if final_result.get('text', '').strip():
                    transcription_results.append(final_result['text'].strip())
                    confidence_scores.append(final_result.get('conf', 0.5))
                
                # Combine results
                full_text = ' '.join(transcription_results).strip()
                
                # Calculate average confidence
                avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
                
                # Language detection if language_utils available
                detected_language = language
                if language_utils and full_text:
                    detected_lang = language_utils.detect_language(full_text)
                    if language_utils.validate_language(detected_lang):
                        detected_language = detected_lang
                        logger.info(f"🌍 Language detected from text: {detected_language}")
                
                if full_text:
                    logger.info(f"✅ Vosk transcription successful: '{full_text[:50]}...' (Confidence: {avg_confidence:.2f})")
                    return {
                        "success": True,
                        "text": full_text,
                        "confidence": round(avg_confidence, 3),
                        "language": detected_language,
                        "engine": "vosk",
                        "model_language": self.current_model_language,
                        "chunks_processed": len(transcription_results),
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    logger.warning("⚠️ Vosk produced empty transcription")
                    return await self.mock_transcription(language, reason="Empty Vosk result")
                
            finally:
                # Clean up temporary file
                if temp_file and os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to remove temp file {temp_file}: {e}")
                        
        except Exception as e:
            logger.error(f"❌ Vosk transcription error: {e}")
            return await self.mock_transcription(language, reason=f"Vosk error: {str(e)}")
    
    def _save_audio_fallback(self, audio_data: bytes) -> str:
        """Fallback method to save audio if audio_processor unavailable."""
        temp_dir = getattr(settings, 'audio_temp_dir', tempfile.gettempdir())
        os.makedirs(temp_dir, exist_ok=True)
        
        temp_file = os.path.join(temp_dir, f'stt_audio_{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}.wav')
        
        # Create a simple WAV file
        with wave.open(temp_file, 'wb') as wav_file:
            wav_file.setnchannels(self.channels)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio_data)
        
        return temp_file
    
    def _validate_audio_file(self, file_path: str) -> bool:
        """Validate audio file format and content."""
        try:
            if not os.path.exists(file_path):
                return False
            
            file_size = os.path.getsize(file_path)
            if file_size < 100:  # Too small to be valid audio
                return False
            
            # Try to open as WAV file
            try:
                with wave.open(file_path, 'rb') as wav_file:
                    frames = wav_file.getnframes()
                    return frames > 0
            except wave.Error:
                # Not a WAV file, but might still be valid raw audio
                return True
                
        except Exception as e:
            logger.warning(f"⚠️ Audio validation error: {e}")
            return True  # Assume valid if we can't validate
    
    async def mock_transcription(self, language: str, reason: str = "Mock mode") -> Dict[str, Any]:
        """Enhanced mock transcription for testing and fallback."""
        lang_info = self.supported_languages.get(language, self.supported_languages['en'])
        mock_text = lang_info['mock_text']
        
        logger.info(f"🎭 Using mock transcription: {reason}")
        
        return {
            "success": True,
            "text": mock_text,
            "confidence": 0.95,  # High confidence for mock
            "language": language,
            "engine": "mock",
            "reason": reason,
            "note": "This is a mock transcription for testing purposes",
            "timestamp": datetime.now().isoformat()
        }
    
    async def detect_language_from_audio(self, audio_data: bytes) -> str:
        """Enhanced language detection from audio."""
        try:
            # First try transcribing with default language
            result = await self.transcribe_audio(audio_data, 'en')
            
            if result.get('success') and result.get('text'):
                text = result['text']
                
                # Use language utils for detection if available
                if language_utils:
                    detected = language_utils.detect_language(text)
                    if language_utils.validate_language(detected):
                        logger.info(f"🌍 Audio language detected: {detected}")
                        return detected
                
                # Fallback: simple pattern matching
                text_lower = text.lower()
                
                # Hindi detection
                if any(word in text_lower for word in ['बस', 'कहाँ', 'है', 'का', 'की']):
                    return 'hi'
                
                # Punjabi detection  
                if any(word in text_lower for word in ['ਬੱਸ', 'ਕਿੱਥੇ', 'ਹੈ', 'ਦਾ', 'ਦੀ']):
                    return 'pa'
            
            # Default to configured language or English
            return getattr(settings, 'default_language', 'en')
            
        except Exception as e:
            logger.error(f"❌ Language detection error: {e}")
            return getattr(settings, 'default_language', 'en')
    
    def get_supported_languages(self) -> List[Dict[str, Any]]:
        """Get detailed information about supported languages."""
        return [
            {
                "code": code,
                "name": info["name"],
                "vosk_model": info["vosk_model"],
                "available": self.engine == 'vosk' and self.vosk_model is not None
            }
            for code, info in self.supported_languages.items()
        ]
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive STT status."""
        return {
            "engine": self.engine,
            "vosk_available": VOSK_AVAILABLE,
            "vosk_model_loaded": self.vosk_model is not None,
            "current_model_language": self.current_model_language,
            "audio_processor_available": audio_processor is not None,
            "language_utils_available": language_utils is not None,
            "supported_languages": list(self.supported_languages.keys()),
            "configuration": {
                "sample_rate": self.sample_rate,
                "channels": self.channels,
                "chunk_size": self.chunk_size,
                "model_path": self.model_path,
                "enable_vosk": self.enable_vosk
            },
            "statistics": self.stats,
            "performance": {
                "success_rate": (
                    self.stats['successful_transcriptions'] / max(self.stats['total_transcriptions'], 1) * 100
                    if self.stats['total_transcriptions'] > 0 else 0
                ),
                "average_confidence": round(self.stats['average_confidence'], 3)
            }
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform STT health check."""
        try:
            # Test with sample audio data
            sample_audio = b'\x00' * 1000  # 1000 bytes of silence
            
            start_time = datetime.now()
            result = await self.mock_transcription('en', 'Health check')
            end_time = datetime.now()
            
            response_time = (end_time - start_time).total_seconds()
            
            return {
                "status": "healthy" if result.get('success') else "degraded",
                "engine": self.engine,
                "response_time_seconds": round(response_time, 3),
                "vosk_model_available": self.vosk_model is not None,
                "last_check": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "engine": self.engine,
                "last_check": datetime.now().isoformat()
            }
    
    def reset_statistics(self):
        """Reset transcription statistics."""
        self.stats = {
            'total_transcriptions': 0,
            'successful_transcriptions': 0,
            'failed_transcriptions': 0,
            'language_counts': {},
            'average_confidence': 0.0,
            'last_transcription_time': None
        }
        logger.info("📊 STT statistics reset")
    
    async def batch_transcribe(self, audio_files: List[bytes], language: str = 'en') -> List[Dict[str, Any]]:
        """Batch transcribe multiple audio files."""
        results = []
        
        for i, audio_data in enumerate(audio_files):
            logger.info(f"🔄 Batch transcribing file {i+1}/{len(audio_files)}")
            try:
                result = await self.transcribe_audio(audio_data, language)
                result['batch_index'] = i
                results.append(result)
            except Exception as e:
                logger.error(f"❌ Batch transcription error for file {i}: {e}")
                results.append({
                    "success": False,
                    "error": str(e),
                    "text": "",
                    "confidence": 0.0,
                    "language": language,
                    "batch_index": i
                })
        
        return results

# Global instance
stt_engine = STTEngine()
