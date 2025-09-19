"""
API Routes - Complete Bus Assistant API
"""
from fastapi import APIRouter, HTTPException, File, UploadFile
from pydantic import BaseModel
from typing import Optional, Dict, Any
import logging

from stt.stt_engine import stt_engine
from tts.tts_engine import tts_engine
from nlu.nlu_engine import nlu_engine
from fetcher.fetch_data import bus_data_fetcher

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
        language = request.preferred_language or "en"
        
        # Generate greeting
        greetings = {
            'en': "Hello! I'm your AI bus assistant. How can I help you find bus information today?",
            'hi': "नमस्ते! मैं आपका AI बस असिस्टेंट हूँ। आज बस की जानकारी के लिए मैं आपकी कैसे मदद कर सकता हूँ?",
            'pa': "ਸਤ ਸ੍ਰੀ ਅਕਾਲ! ਮੈਂ ਤੁਹਾਡਾ AI ਬੱਸ ਅਸਿਸਟੈਂਟ ਹਾਂ। ਅੱਜ ਬੱਸ ਦੀ ਜਾਣਕਾਰੀ ਲਈ ਮੈਂ ਤੁਹਾਡੀ ਕਿਵੇਂ ਮਦਦ ਕਰ ਸਕਦਾ ਹਾਂ?"
        }
        
        greeting_text = greetings.get(language, greetings['en'])
        
        # Generate greeting audio
        tts_result = await tts_engine.text_to_speech(greeting_text, language)
        
        return {
            "success": True,
            "greeting_text": greeting_text,
            "greeting_audio_url": tts_result.get('audio_url'),
            "detected_language": language,
            "supported_languages": ["en", "hi", "pa"],
            "session_id": request.client_id
        }
        
    except Exception as e:
        logger.error(f"Conversation start error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/voice/query")
async def process_voice_query(request: VoiceQuery) -> Dict[str, Any]:
    """Process text-based voice query."""
    try:
        # Process query with NLU
        nlu_result = await nlu_engine.process_query(request.text, request.language)
        
        if not nlu_result.get('success'):
            raise HTTPException(status_code=400, detail=nlu_result.get('error', 'NLU processing failed'))
        
        response_text = nlu_result['response']
        
        # Generate audio response
        tts_result = await tts_engine.text_to_speech(response_text, request.language)
        
        return {
            "success": True,
            "query_text": request.text,
            "response_text": response_text,
            "detected_language": request.language,
            "intent": nlu_result.get('intent'),
            "audio_url": tts_result.get('audio_url'),
            "data": nlu_result.get('data'),
            "processing_time": 1.5  # Mock processing time
        }
        
    except Exception as e:
        logger.error(f"Voice query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/voice/process")
async def process_voice_file(audio: UploadFile = File(...), language: Optional[str] = "en") -> Dict[str, Any]:
    """Process uploaded audio file."""
    try:
        # Read audio data
        audio_data = await audio.read()
        
        if len(audio_data) < 100:
            raise HTTPException(status_code=400, detail="Audio file too small")
        
        # Transcribe audio
        stt_result = await stt_engine.transcribe_audio(audio_data, language)
        
        if not stt_result.get('success'):
            raise HTTPException(status_code=400, detail="Speech recognition failed")
        
        transcribed_text = stt_result['text']
        detected_language = stt_result.get('language', language)
        
        # Process with NLU
        nlu_result = await nlu_engine.process_query(transcribed_text, detected_language)
        
        if not nlu_result.get('success'):
            raise HTTPException(status_code=400, detail="Natural language understanding failed")
        
        response_text = nlu_result['response']
        
        # Generate audio response
        tts_result = await tts_engine.text_to_speech(response_text, detected_language)
        
        return {
            "success": True,
            "transcription": {
                "text": transcribed_text,
                "detected_language": detected_language,
                "confidence": stt_result.get('confidence', 0.0)
            },
            "response": {
                "text": response_text,
                "audio_url": tts_result.get('audio_url'),
                "intent": nlu_result.get('intent')
            },
            "data": nlu_result.get('data'),
            "processing_time": 2.1
        }
        
    except Exception as e:
        logger.error(f"Voice processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/voice/languages")
async def get_supported_languages() -> Dict[str, Any]:
    """Get supported languages and voices."""
    return {
        "supported_languages": [
            {
                "code": "en",
                "name": "English",
                "voice": "en-US-AriaNeural"
            },
            {
                "code": "hi", 
                "name": "Hindi",
                "voice": "hi-IN-SwaraNeural"
            },
            {
                "code": "pa",
                "name": "Punjabi", 
                "voice": "pa-IN-GulNeural"
            }
        ],
        "default_language": "en",
        "auto_detection": True
    }

@router.get("/bus/search")
async def search_buses(start: Optional[str] = None, end: Optional[str] = None) -> Dict[str, Any]:
    """Search buses by route."""
    try:
        result = await bus_data_fetcher.search_buses(start, end)
        return result
    except Exception as e:
        logger.error(f"Bus search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/bus/{bus_id}")
async def get_bus_details(bus_id: str) -> Dict[str, Any]:
    """Get bus details by ID."""
    try:
        result = await bus_data_fetcher.get_bus_details(bus_id)
        return result
    except Exception as e:
        logger.error(f"Bus details error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/bus/by-name/{name}")
async def get_bus_by_name(name: str) -> Dict[str, Any]:
    """Get bus by name."""
    try:
        result = await bus_data_fetcher.get_bus_by_name(name)
        return result
    except Exception as e:
        logger.error(f"Bus by name error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status")
async def get_system_status() -> Dict[str, Any]:
    """Get system status."""
    try:
        return {
            "service": "AI Bus Assistant API",
            "status": "healthy",
            "components": {
                "stt": stt_engine.get_status(),
                "tts": tts_engine.get_status(),
                "nlu": nlu_engine.get_status(),
                "bus_fetcher": {
                    "backend_url": bus_data_fetcher.base_url,
                    "cache_duration": bus_data_fetcher.cache_duration
                }
            }
        }
    except Exception as e:
        logger.error(f"Status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
