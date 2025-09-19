"""
TTS Engine - Production Ready with Enhanced Features & Resource Management
Compatible with Python 3.13 and Render deployment
"""
import asyncio
import os
import uuid
import shutil
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime, timedelta
from config.settings import settings

# Import language utils for voice selection
try:
    from utils.language_utils import language_utils
except ImportError:
    language_utils = None

try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False

logger = logging.getLogger(__name__)

class TTSEngine:
    """Enhanced Text-to-Speech engine with production-grade features."""
    
    def __init__(self):
        # Engine settings
        self.engine = getattr(settings, 'tts_engine', 'edge_tts')
        self.temp_dir = getattr(settings, 'audio_temp_dir', 'temp/audio')
        self.static_audio_dir = "static/audio"
        
        # Enhanced voice configuration with quality and gender options
        self.voices = {
            'en': {
                'primary': getattr(settings, 'tts_voice_en', 'en-US-AriaNeural'),
                'alternatives': [
                    'en-US-JennyNeural',  # Female, friendly
                    'en-US-GuyNeural',    # Male, professional
                    'en-US-AriaNeural',   # Female, neutral
                    'en-US-DavisNeural'   # Male, conversational
                ],
                'language_name': 'English'
            },
            'hi': {
                'primary': getattr(settings, 'tts_voice_hi', 'hi-IN-SwaraNeural'),
                'alternatives': [
                    'hi-IN-SwaraNeural',  # Female, clear
                    'hi-IN-MadhurNeural', # Male, warm
                    'hi-IN-AaravNeural'   # Male, professional
                ],
                'language_name': 'Hindi'
            },
            'pa': {
                'primary': getattr(settings, 'tts_voice_pa', 'pa-IN-GulNeural'),
                'alternatives': [
                    'pa-IN-GulNeural',    # Female, clear
                    'pa-IN-HarpreetNeural' # Male, natural
                ],
                'language_name': 'Punjabi'
            }
        }
        
        # Audio file management
        self.max_file_age = timedelta(hours=24)  # Clean files older than 24 hours
        self.max_cache_size_mb = 500  # Maximum cache size in MB
        
        # Statistics
        self.stats = {
            'total_synthesis': 0,
            'successful_synthesis': 0,
            'failed_synthesis': 0,
            'language_counts': {},
            'cache_hits': 0,
            'total_characters': 0,
            'average_synthesis_time': 0.0,
            'last_synthesis_time': None
        }
        
        # Simple cache for recent syntheses
        self.synthesis_cache = {}
        self.cache_max_size = 50
        
        # Initialize directories and engine
        self.initialize_directories()
        self.initialize_engine()
        
        logger.info(f"✅ TTSEngine initialized - Engine: {self.engine}")
        logger.info(f"📁 Audio directories: temp={self.temp_dir}, static={self.static_audio_dir}")
    
    def initialize_directories(self):
        """Initialize and validate audio directories."""
        try:
            # Create directories
            os.makedirs(self.temp_dir, exist_ok=True)
            os.makedirs(self.static_audio_dir, exist_ok=True)
            
            # Verify write permissions
            test_file = os.path.join(self.temp_dir, 'test_write.tmp')
            try:
                with open(test_file, 'w') as f:
                    f.write('test')
                os.remove(test_file)
                logger.info("✅ Audio directory permissions verified")
            except Exception as e:
                logger.error(f"❌ Audio directory write test failed: {e}")
                
        except Exception as e:
            logger.error(f"❌ Directory initialization error: {e}")
    
    def initialize_engine(self):
        """Initialize TTS engine with availability checking."""
        if self.engine == 'edge_tts' and EDGE_TTS_AVAILABLE:
            logger.info("✅ Edge TTS available and configured")
            # Test Edge TTS availability
            logger.info("✅ Edge TTS configured - will test on first synthesis")

            #asyncio.create_task(self.test_edge_tts())
        else:
            self.engine = 'mock'
            logger.warning("⚠️ Edge TTS not available - using mock TTS")
            if not EDGE_TTS_AVAILABLE:
                logger.info("💡 To enable Edge TTS: pip install edge-tts")
    
    async def test_edge_tts(self):
        """Test Edge TTS availability and voice access."""
        try:
            # Test with a simple synthesis
            test_text = "Test"
            voice = self.voices['en']['primary']
            
            communicate = edge_tts.Communicate(test_text, voice)
            
            # Try to get voice data (but don't save)
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    logger.info(f"✅ Edge TTS voice test successful: {voice}")
                    break
                    
        except Exception as e:
            logger.warning(f"⚠️ Edge TTS test failed, falling back to mock: {e}")
            self.engine = 'mock'
    
    async def text_to_speech(self, text: str, language: str = 'en', voice: str = None) -> Dict[str, Any]:
        """Enhanced text-to-speech with caching and error handling."""
        try:
            # Update statistics
            self.stats['total_synthesis'] += 1
            self.stats['total_characters'] += len(text)
            self.stats['last_synthesis_time'] = datetime.now().isoformat()
            self.stats['language_counts'][language] = self.stats['language_counts'].get(language, 0) + 1
            
            start_time = datetime.now()
            
            # Validate input
            if not text or not text.strip():
                raise ValueError("Text cannot be empty")
            
            text = text.strip()
            
            # Validate and normalize language
            if language_utils:
                language = language_utils.normalize_language_code(language)
                if not language_utils.validate_language(language):
                    logger.warning(f"⚠️ Unsupported language {language}, using English")
                    language = 'en'
            elif language not in self.voices:
                logger.warning(f"⚠️ Unsupported language {language}, using English")
                language = 'en'
            
            logger.info(f"🎵 TTS synthesis: '{text[:50]}...' in {language}")
            
            # Check cache first
            cache_key = f"{language}:{text[:100]}:{voice}"
            cached_result = self.get_cached_result(cache_key)
            if cached_result:
                self.stats['cache_hits'] += 1
                logger.info(f"🔄 TTS cache hit")
                return cached_result
            
            # Perform synthesis
            if self.engine == 'edge_tts' and EDGE_TTS_AVAILABLE:
                result = await self.edge_tts_synthesis(text, language, voice)
            else:
                result = await self.mock_tts(text, language, voice)
            
            # Update statistics
            end_time = datetime.now()
            synthesis_time = (end_time - start_time).total_seconds()
            
            # Update average synthesis time
            total_successful = self.stats['successful_synthesis']
            current_avg = self.stats['average_synthesis_time']
            self.stats['average_synthesis_time'] = (current_avg * total_successful + synthesis_time) / (total_successful + 1)
            
            if result.get('success'):
                self.stats['successful_synthesis'] += 1
                
                # Cache successful result
                self.cache_result(cache_key, result)
                
                # Add synthesis time to result
                result['synthesis_time_seconds'] = round(synthesis_time, 3)
            else:
                self.stats['failed_synthesis'] += 1
            
            # Cleanup old files periodically
            if self.stats['total_synthesis'] % 10 == 0:
                await self.cleanup_old_files()
            
            return result
            
        except Exception as e:
            logger.error(f"❌ TTS synthesis error: {e}")
            self.stats['failed_synthesis'] += 1
            
            return {
                "success": False,
                "error": str(e),
                "text": text,
                "language": language,
                "audio_file": None,
                "audio_url": None,
                "engine": self.engine,
                "timestamp": datetime.now().isoformat()
            }
    
    async def edge_tts_synthesis(self, text: str, language: str, voice: str = None) -> Dict[str, Any]:
        """Enhanced Edge TTS synthesis with voice selection and error handling."""
        try:
            # Select voice
            selected_voice = self.select_voice(language, voice)
            
            # Generate unique filename
            filename = f"tts_{language}_{uuid.uuid4().hex[:12]}.wav"
            temp_filepath = os.path.join(self.temp_dir, filename)
            static_filepath = os.path.join(self.static_audio_dir, filename)
            
            logger.info(f"🎤 Generating TTS with voice: {selected_voice}")
            
            # Generate speech using Edge TTS
            communicate = edge_tts.Communicate(text, selected_voice)
            
            try:
                await communicate.save(temp_filepath)
                
                # Validate generated file
                if not os.path.exists(temp_filepath) or os.path.getsize(temp_filepath) < 100:
                    raise Exception("Generated audio file is invalid or too small")
                
                # Move to static directory for serving
                shutil.move(temp_filepath, static_filepath)
                
                # Create URL for serving
                audio_url = f"/static/audio/{filename}"
                
                logger.info(f"✅ TTS synthesis successful: {filename}")
                
                return {
                    "success": True,
                    "text": text,
                    "language": language,
                    "voice": selected_voice,
                    "audio_file": static_filepath,
                    "audio_url": audio_url,
                    "engine": "edge_tts",
                    "file_size_bytes": os.path.getsize(static_filepath),
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                # Clean up temp file if it exists
                if os.path.exists(temp_filepath):
                    os.remove(temp_filepath)
                raise e
                
        except Exception as e:
            logger.error(f"❌ Edge TTS synthesis error: {e}")
            
            # Fallback to mock TTS
            return await self.mock_tts(text, language, voice, f"Edge TTS failed: {str(e)}")
    
    def select_voice(self, language: str, requested_voice: str = None) -> str:
        """Intelligent voice selection based on language and preferences."""
        lang_config = self.voices.get(language, self.voices['en'])
        
        # Use requested voice if provided and valid
        if requested_voice:
            all_voices = [lang_config['primary']] + lang_config.get('alternatives', [])
            if requested_voice in all_voices:
                return requested_voice
            else:
                logger.warning(f"⚠️ Requested voice '{requested_voice}' not available, using default")
        
        # Use language utils for voice recommendation if available
        if language_utils:
            lang_info = language_utils.get_language_info(language)
            recommended_voice = lang_info.get('tts_voice')
            if recommended_voice and recommended_voice in lang_config.get('alternatives', []):
                return recommended_voice
        
        # Return primary voice for language
        return lang_config['primary']
    
    async def mock_tts(self, text: str, language: str, voice: str = None, reason: str = "Mock mode") -> Dict[str, Any]:
        """Enhanced mock TTS for testing and fallback."""
        filename = f"mock_tts_{language}_{uuid.uuid4().hex[:8]}.wav"
        audio_url = f"/static/audio/{filename}"
        
        # Create a simple mock audio file indicator
        mock_filepath = os.path.join(self.static_audio_dir, f"{filename}.txt")
        try:
            with open(mock_filepath, 'w') as f:
                f.write(f"Mock TTS audio for: {text[:100]}\nLanguage: {language}\nVoice: {voice}\nGenerated: {datetime.now().isoformat()}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to create mock file: {e}")
        
        logger.info(f"🎭 Mock TTS: {reason}")
        
        return {
            "success": True,
            "text": text,
            "language": language,
            "voice": voice or "mock-voice",
            "audio_file": mock_filepath if os.path.exists(mock_filepath) else None,
            "audio_url": audio_url,
            "engine": "mock",
            "reason": reason,
            "note": "Mock TTS - no actual audio generated",
            "mock": True,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_cached_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached synthesis result if available and valid."""
        if cache_key in self.synthesis_cache:
            cached_item = self.synthesis_cache[cache_key]
            
            # Check if cache is still valid (file exists)
            if 'audio_file' in cached_item and cached_item['audio_file']:
                if os.path.exists(cached_item['audio_file']):
                    return cached_item
                else:
                    # Remove invalid cache entry
                    del self.synthesis_cache[cache_key]
        
        return None
    
    def cache_result(self, cache_key: str, result: Dict[str, Any]):
        """Cache synthesis result with size management."""
        if result.get('success') and not result.get('mock'):
            # Limit cache size
            if len(self.synthesis_cache) >= self.cache_max_size:
                # Remove oldest entry
                oldest_key = next(iter(self.synthesis_cache))
                del self.synthesis_cache[oldest_key]
            
            self.synthesis_cache[cache_key] = result.copy()
    
    async def cleanup_old_files(self):
        """Clean up old audio files to manage storage."""
        try:
            current_time = datetime.now()
            cleaned_count = 0
            total_size_freed = 0
            
            # Clean static audio directory
            for filename in os.listdir(self.static_audio_dir):
                file_path = os.path.join(self.static_audio_dir, filename)
                
                if os.path.isfile(file_path):
                    # Check file age
                    file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                    if current_time - file_time > self.max_file_age:
                        try:
                            file_size = os.path.getsize(file_path)
                            os.remove(file_path)
                            cleaned_count += 1
                            total_size_freed += file_size
                            logger.debug(f"🗑️ Cleaned old TTS file: {filename}")
                        except Exception as e:
                            logger.warning(f"⚠️ Failed to clean file {filename}: {e}")
            
            # Clean temp directory
            for filename in os.listdir(self.temp_dir):
                file_path = os.path.join(self.temp_dir, filename)
                
                if os.path.isfile(file_path):
                    file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                    if current_time - file_time > timedelta(hours=1):  # Clean temp files after 1 hour
                        try:
                            file_size = os.path.getsize(file_path)
                            os.remove(file_path)
                            cleaned_count += 1
                            total_size_freed += file_size
                        except Exception as e:
                            logger.warning(f"⚠️ Failed to clean temp file {filename}: {e}")
            
            if cleaned_count > 0:
                logger.info(f"🧹 Cleaned {cleaned_count} old TTS files, freed {total_size_freed/1024/1024:.1f} MB")
                
        except Exception as e:
            logger.error(f"❌ File cleanup error: {e}")
    
    async def batch_synthesis(self, texts: List[str], language: str = 'en', voice: str = None) -> List[Dict[str, Any]]:
        """Batch text-to-speech synthesis."""
        results = []
        
        logger.info(f"🔄 Starting batch TTS synthesis: {len(texts)} texts in {language}")
        
        for i, text in enumerate(texts):
            try:
                result = await self.text_to_speech(text, language, voice)
                result['batch_index'] = i
                results.append(result)
                
                # Small delay to prevent overwhelming the service
                if i < len(texts) - 1:
                    await asyncio.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"❌ Batch synthesis error for text {i}: {e}")
                results.append({
                    "success": False,
                    "error": str(e),
                    "text": text,
                    "language": language,
                    "batch_index": i
                })
        
        logger.info(f"✅ Batch synthesis completed: {len(results)} results")
        return results
    
    def get_available_voices(self, language: str = None) -> Dict[str, Any]:
        """Get available voices for specified language or all languages."""
        if language:
            if language in self.voices:
                return {
                    language: {
                        "language_name": self.voices[language]['language_name'],
                        "primary_voice": self.voices[language]['primary'],
                        "alternative_voices": self.voices[language].get('alternatives', []),
                        "total_voices": 1 + len(self.voices[language].get('alternatives', []))
                    }
                }
            else:
                return {}
        else:
            return {
                lang: {
                    "language_name": config['language_name'],
                    "primary_voice": config['primary'],
                    "alternative_voices": config.get('alternatives', []),
                    "total_voices": 1 + len(config.get('alternatives', []))
                }
                for lang, config in self.voices.items()
            }
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive TTS status."""
        # Calculate cache and storage info
        cache_info = {
            "cache_size": len(self.synthesis_cache),
            "cache_max_size": self.cache_max_size,
            "cache_hit_rate": (
                self.stats['cache_hits'] / max(self.stats['total_synthesis'], 1) * 100
                if self.stats['total_synthesis'] > 0 else 0
            )
        }
        
        # Calculate storage info
        storage_info = {}
        try:
            static_size = sum(
                os.path.getsize(os.path.join(self.static_audio_dir, f))
                for f in os.listdir(self.static_audio_dir)
                if os.path.isfile(os.path.join(self.static_audio_dir, f))
            )
            temp_size = sum(
                os.path.getsize(os.path.join(self.temp_dir, f))
                for f in os.listdir(self.temp_dir)
                if os.path.isfile(os.path.join(self.temp_dir, f))
            )
            
            storage_info = {
                "static_directory_size_mb": round(static_size / 1024 / 1024, 2),
                "temp_directory_size_mb": round(temp_size / 1024 / 1024, 2),
                "total_size_mb": round((static_size + temp_size) / 1024 / 1024, 2)
            }
        except Exception as e:
            storage_info = {"error": f"Could not calculate storage: {e}"}
        
        return {
            "engine": self.engine,
            "edge_tts_available": EDGE_TTS_AVAILABLE,
            "language_utils_available": language_utils is not None,
            "supported_languages": list(self.voices.keys()),
            "total_voices": sum(1 + len(config.get('alternatives', [])) for config in self.voices.values()),
            "configuration": {
                "temp_directory": self.temp_dir,
                "static_directory": self.static_audio_dir,
                "max_file_age_hours": self.max_file_age.total_seconds() / 3600,
                "max_cache_size_mb": self.max_cache_size_mb
            },
            "statistics": self.stats,
            "cache_info": cache_info,
            "storage_info": storage_info,
            "performance": {
                "success_rate": (
                    self.stats['successful_synthesis'] / max(self.stats['total_synthesis'], 1) * 100
                    if self.stats['total_synthesis'] > 0 else 0
                ),
                "average_synthesis_time_seconds": round(self.stats['average_synthesis_time'], 3),
                "characters_per_synthesis": (
                    self.stats['total_characters'] / max(self.stats['total_synthesis'], 1)
                    if self.stats['total_synthesis'] > 0 else 0
                )
            }
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform TTS health check."""
        try:
            # Test synthesis
            test_text = "Health check test"
            start_time = datetime.now()
            
            if self.engine == 'edge_tts':
                # Test actual Edge TTS
                result = await self.text_to_speech(test_text, 'en')
            else:
                # Test mock TTS
                result = await self.mock_tts(test_text, 'en', reason='Health check')
            
            end_time = datetime.now()
            response_time = (end_time - start_time).total_seconds()
            
            return {
                "status": "healthy" if result.get('success') else "degraded",
                "engine": self.engine,
                "response_time_seconds": round(response_time, 3),
                "test_synthesis_success": result.get('success', False),
                "directories_writable": os.access(self.temp_dir, os.W_OK) and os.access(self.static_audio_dir, os.W_OK),
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
        """Reset TTS statistics and clear cache."""
        self.stats = {
            'total_synthesis': 0,
            'successful_synthesis': 0,
            'failed_synthesis': 0,
            'language_counts': {},
            'cache_hits': 0,
            'total_characters': 0,
            'average_synthesis_time': 0.0,
            'last_synthesis_time': None
        }
        self.synthesis_cache.clear()
        logger.info("📊 TTS statistics and cache reset")

# Global instance
tts_engine = TTSEngine()
