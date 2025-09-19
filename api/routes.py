"""
API Routes - Production Ready with Language Utils Integration
"""
from fastapi import APIRouter, HTTPException, File, UploadFile
from pydantic import BaseModel
from typing import Optional, Dict, Any
import logging
from datetime import datetime

# Core engine imports with error handling
try:
    from stt.stt_engine import stt_engine
except ImportError as e:
    print(f"⚠️ STT engine import failed: {e}")
    stt_engine = None

try:
    from tts.tts_engine import tts_engine
except ImportError as e:
    print(f"⚠️ TTS engine import failed: {e}")
    tts_engine = None

try:
    from nlu.nlu_engine import nlu_engine
except ImportError as e:
    print(f"⚠️ NLU engine import failed: {e}")
    nlu_engine = None

try:
    from fetcher.fetch_data import bus_data_fetcher
except ImportError as e:
    print(f"⚠️ Bus data fetcher import failed: {e}")
    bus_data_fetcher = None

try:
    from utils.language_utils import language_utils
except ImportError as e:
    print(f"⚠️ Language utils import failed: {e}")
    language_utils = None

logger = logging.getLogger(__name__)
router = APIRouter()

# Request models
class ConversationStart(BaseModel):
    client_id: str
    preferred_language: Optional[str] = "en"

class VoiceQuery(BaseModel):
    text: str
    language: Optional[str] = "en"
    detect_language: bool = True

class BusSearch(BaseModel):
    start: Optional[str] = None
    end: Optional[str] = None

# Response models
class ConversationResponse(BaseModel):
    success: bool
    greeting_text: str
    greeting_audio_url: Optional[str] = None
    detected_language: str
    supported_languages: list

@router.post("/conversation/start")
async def start_conversation(request: ConversationStart) -> Dict[str, Any]:
    """Start a new conversation with greeting."""
    try:
        # Normalize language
        language = request.preferred_language or "en"
        if language_utils:
            language = language_utils.normalize_language_code(language)
        
        # Generate greeting
        greetings = {
            'en': "Hello! I'm your AI bus assistant. How can I help you find bus information today?",
            'hi': "नमस्ते! मैं आपका AI बस असिस्टेंट हूँ। आज बस की जानकारी के लिए मैं आपकी कैसे मदद कर सकता हूँ?",
            'pa': "ਸਤ ਸ੍ਰੀ ਅਕਾਲ! ਮੈਂ ਤੁਹਾਡਾ AI ਬੱਸ ਅਸਿਸਟੈਂਟ ਹਾਂ। ਅੱਜ ਬੱਸ ਦੀ ਜਾਣਕਾਰੀ ਲਈ ਮੈਂ ਤੁਹਾਡੀ ਕਿਵੇਂ ਮਦਦ ਕਰ ਸਕਦਾ ਹਾਂ?"
        }
        
        greeting_text = greetings.get(language, greetings['en'])
        
        # Generate greeting audio
        greeting_audio_url = None
        if tts_engine:
            try:
                tts_result = await tts_engine.text_to_speech(greeting_text, language)
                greeting_audio_url = tts_result.get('audio_url')
            except Exception as e:
                logger.warning(f"TTS greeting failed: {e}")
        
        # Get supported languages
        supported_languages = ["en", "hi", "pa"]
        if language_utils:
            supported_languages = language_utils.supported_languages
        
        return {
            "success": True,
            "greeting_text": greeting_text,
            "greeting_audio_url": greeting_audio_url,
            "detected_language": language,
            "supported_languages": supported_languages,
            "session_id": request.client_id,
            "language_info": language_utils.get_language_info(language) if language_utils else None
        }
        
    except Exception as e:
        logger.error(f"Conversation start error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/voice/query")
async def process_voice_query(request: VoiceQuery) -> Dict[str, Any]:
    """Process text-based voice query with enhanced language detection."""
    try:
        # Auto-detect language if enabled
        detected_language = request.language or "en"
        
        if request.detect_language and language_utils:
            detected_language = language_utils.detect_language(request.text)
            logger.info(f"🌍 Language detected: '{request.text[:30]}...' → {detected_language}")
        
        # Validate and normalize language
        if language_utils:
            if not language_utils.validate_language(detected_language):
                detected_language = language_utils.default_language
        
        # Check NLU availability
        if not nlu_engine:
            raise HTTPException(status_code=503, detail="NLU engine not available")
        
        # Process query with NLU
        nlu_result = await nlu_engine.process_query(request.text, detected_language)
        
        if not nlu_result.get('success'):
            raise HTTPException(status_code=400, detail=nlu_result.get('error', 'NLU processing failed'))
        
        response_text = nlu_result['response']
        
        # Generate audio response with correct voice
        audio_url = None
        if tts_engine:
            try:
                tts_result = await tts_engine.text_to_speech(response_text, detected_language)
                audio_url = tts_result.get('audio_url')
            except Exception as e:
                logger.warning(f"TTS generation failed: {e}")
        
        return {
            "success": True,
            "query_text": request.text,
            "response_text": response_text,
            "detected_language": detected_language,
            "language_info": language_utils.get_language_info(detected_language) if language_utils else None,
            "intent": nlu_result.get('intent'),
            "audio_url": audio_url,
            "data": nlu_result.get('data'),
            "processing_time": 1.5,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Voice query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/voice/process")
async def process_voice_file(audio: UploadFile = File(...), language: Optional[str] = "en") -> Dict[str, Any]:
    """Process uploaded audio file with enhanced error handling."""
    try:
        # Check STT availability
        if not stt_engine:
            raise HTTPException(status_code=503, detail="Speech recognition not available")
        
        # Read and validate audio data
        audio_data = await audio.read()
        
        if len(audio_data) < 100:
            raise HTTPException(status_code=400, detail="Audio file too small")
        
        # Transcribe audio
        stt_result = await stt_engine.transcribe_audio(audio_data, language)
        
        if not stt_result.get('success'):
            raise HTTPException(status_code=400, detail="Speech recognition failed")
        
        transcribed_text = stt_result['text']
        detected_language = stt_result.get('language', language)
        
        # Auto-detect language from transcribed text if language utils available
        if language_utils and transcribed_text:
            text_language = language_utils.detect_language(transcribed_text)
            if language_utils.validate_language(text_language):
                detected_language = text_language
                logger.info(f"🌍 Text-based language detection: {detected_language}")
        
        # Check NLU availability
        if not nlu_engine:
            raise HTTPException(status_code=503, detail="NLU engine not available")
        
        # Process with NLU
        nlu_result = await nlu_engine.process_query(transcribed_text, detected_language)
        
        if not nlu_result.get('success'):
            raise HTTPException(status_code=400, detail="Natural language understanding failed")
        
        response_text = nlu_result['response']
        
        # Generate audio response
        audio_url = None
        if tts_engine:
            try:
                tts_result = await tts_engine.text_to_speech(response_text, detected_language)
                audio_url = tts_result.get('audio_url')
            except Exception as e:
                logger.warning(f"TTS generation failed: {e}")
        
        return {
            "success": True,
            "transcription": {
                "text": transcribed_text,
                "detected_language": detected_language,
                "confidence": stt_result.get('confidence', 0.0),
                "language_info": language_utils.get_language_info(detected_language) if language_utils else None
            },
            "response": {
                "text": response_text,
                "audio_url": audio_url,
                "intent": nlu_result.get('intent')
            },
            "data": nlu_result.get('data'),
            "processing_time": 2.1,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Voice processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/voice/languages")
async def get_supported_languages() -> Dict[str, Any]:
    """Get comprehensive supported languages information."""
    try:
        if language_utils:
            # Use language utils for comprehensive info
            return {
                "supported_languages": [
                    {
                        "code": lang,
                        "name": language_utils.get_language_name(lang),
                        "native_name": language_utils.get_language_name(lang, native=True),
                        "voice": language_utils.get_tts_voice(lang),
                        "script": language_utils.get_language_info(lang).get('script'),
                        "locale": language_utils.get_language_info(lang).get('locale')
                    }
                    for lang in language_utils.supported_languages
                ],
                "default_language": language_utils.default_language,
                "auto_detection_enabled": True,
                "detection_methods": ["script_based", "pattern_matching", "keyword_analysis", "gemini_ai"],
                "total_supported": len(language_utils.supported_languages)
            }
        else:
            # Fallback if language_utils not available
            return {
                "supported_languages": [
                    {
                        "code": "en",
                        "name": "English",
                        "native_name": "English",
                        "voice": "en-US-AriaNeural",
                        "script": "latin",
                        "locale": "en-US"
                    },
                    {
                        "code": "hi", 
                        "name": "Hindi",
                        "native_name": "हिंदी",
                        "voice": "hi-IN-SwaraNeural",
                        "script": "devanagari",
                        "locale": "hi-IN"
                    },
                    {
                        "code": "pa",
                        "name": "Punjabi", 
                        "native_name": "ਪੰਜਾਬੀ",
                        "voice": "pa-IN-GulNeural",
                        "script": "gurmukhi",
                        "locale": "pa-IN"
                    }
                ],
                "default_language": "en",
                "auto_detection_enabled": False,
                "detection_methods": [],
                "total_supported": 3,
                "note": "Basic language support (language_utils not available)"
            }
    except Exception as e:
        logger.error(f"Language info error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/bus/search")
async def search_buses(start: Optional[str] = None, end: Optional[str] = None) -> Dict[str, Any]:
    """Search buses by route with error handling."""
    try:
        if not bus_data_fetcher:
            raise HTTPException(status_code=503, detail="Bus data service not available")
        
        result = await bus_data_fetcher.search_buses(start, end)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Bus search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/bus/{bus_id}")
async def get_bus_details(bus_id: str) -> Dict[str, Any]:
    """Get bus details by ID with error handling."""
    try:
        if not bus_data_fetcher:
            raise HTTPException(status_code=503, detail="Bus data service not available")
        
        result = await bus_data_fetcher.get_bus_details(bus_id)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Bus details error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/bus/by-name/{name}")
async def get_bus_by_name(name: str) -> Dict[str, Any]:
    """Get bus by name with error handling."""
    try:
        if not bus_data_fetcher:
            raise HTTPException(status_code=503, detail="Bus data service not available")
        
        result = await bus_data_fetcher.get_bus_by_name(name)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Bus by name error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """Detailed health check for all components."""
    components = {}
    overall_status = "healthy"
    error_count = 0
    
    # Check STT engine
    if stt_engine:
        try:
            components["stt"] = stt_engine.get_status()
            components["stt"]["available"] = True
        except Exception as e:
            components["stt"] = {"status": "error", "error": str(e), "available": False}
            error_count += 1
    else:
        components["stt"] = {"status": "not_available", "available": False}
        error_count += 1
    
    # Check TTS engine
    if tts_engine:
        try:
            components["tts"] = tts_engine.get_status()
            components["tts"]["available"] = True
        except Exception as e:
            components["tts"] = {"status": "error", "error": str(e), "available": False}
            error_count += 1
    else:
        components["tts"] = {"status": "not_available", "available": False}
        error_count += 1
    
    # Check NLU engine
    if nlu_engine:
        try:
            components["nlu"] = nlu_engine.get_status()
            components["nlu"]["available"] = True
        except Exception as e:
            components["nlu"] = {"status": "error", "error": str(e), "available": False}
            error_count += 1
    else:
        components["nlu"] = {"status": "not_available", "available": False}
        error_count += 1
    
    # Check bus data fetcher
    if bus_data_fetcher:
        try:
            components["bus_fetcher"] = {
                "status": "healthy",
                "available": True,
                "backend_url": bus_data_fetcher.base_url,
                "cache_duration": bus_data_fetcher.cache_duration
            }
        except Exception as e:
            components["bus_fetcher"] = {"status": "error", "error": str(e), "available": False}
            error_count += 1
    else:
        components["bus_fetcher"] = {"status": "not_available", "available": False}
        error_count += 1
    
    # Check language utils
    if language_utils:
        try:
            components["language_utils"] = {
                "status": "healthy",
                "available": True,
                "supported_languages": language_utils.supported_languages,
                "default_language": language_utils.default_language,
                "detection_enabled": True
            }
        except Exception as e:
            components["language_utils"] = {"status": "error", "error": str(e), "available": False}
    else:
        components["language_utils"] = {
            "status": "not_available", 
            "available": False,
            "detection_enabled": False
        }
    
    # Determine overall status
    if error_count > 2:
        overall_status = "unhealthy"
    elif error_count > 0:
        overall_status = "degraded"
    
    return {
        "status": overall_status,
        "service": "AI Bus Assistant API",
        "timestamp": datetime.now().isoformat(),
        "components": components,
        "summary": {
            "total_components": len(components),
            "available_components": sum(1 for comp in components.values() if comp.get("available", False)),
            "error_count": error_count
        },
        "api_info": {
            "endpoints_available": True,
            "voice_processing": stt_engine is not None and tts_engine is not None and nlu_engine is not None,
            "bus_data": bus_data_fetcher is not None,
            "language_detection": language_utils is not None
        }
    }

@router.get("/status")
async def get_system_status() -> Dict[str, Any]:
    """Get system status (legacy endpoint)."""
    try:
        components = {}
        
        if stt_engine:
            try:
                components["stt"] = stt_engine.get_status()
            except Exception as e:
                components["stt"] = {"status": "error", "error": str(e)}
        
        if tts_engine:
            try:
                components["tts"] = tts_engine.get_status()
            except Exception as e:
                components["tts"] = {"status": "error", "error": str(e)}
        
        if nlu_engine:
            try:
                components["nlu"] = nlu_engine.get_status()
            except Exception as e:
                components["nlu"] = {"status": "error", "error": str(e)}
        
        if bus_data_fetcher:
            components["bus_fetcher"] = {
                "backend_url": bus_data_fetcher.base_url,
                "cache_duration": bus_data_fetcher.cache_duration
            }
        
        return {
            "service": "AI Bus Assistant API",
            "status": "healthy",
            "components": components,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
