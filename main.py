"""
AI Bus Assistant - Backend API (Complete with Language Utils)
"""
import uvicorn
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from config.settings import settings
from api.routes import router as api_router

# Production logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🚀 AI Bus Assistant Backend API Starting")
    
    # Create directories
    directories = ["static/audio", settings.audio_temp_dir, "temp/cache", "logs", "models"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.debug(f"📁 Directory ensured: {directory}")
    
    # Initialize components in order
    components_status = {}
    
    try:
        from utils.language_utils import language_utils
        logger.info(f"✅ Language utilities ready - Supported: {language_utils.supported_languages}")
        components_status['language_utils'] = 'ready'
    except Exception as e:
        logger.error(f"❌ Language utilities error: {e}")
        components_status['language_utils'] = 'error'
    
    try:
        from utils.audio_utils import audio_processor
        logger.info("✅ Audio processor initialized")
        components_status['audio_processor'] = 'ready'
    except Exception as e:
        logger.warning(f"⚠️ Audio processor: {e}")
        components_status['audio_processor'] = 'warning'
    
    try:
        from fetcher.fetch_data import bus_data_fetcher
        await bus_data_fetcher.initialize_session()
        logger.info("✅ Bus data fetcher connected")
        components_status['bus_fetcher'] = 'ready'
    except Exception as e:
        logger.warning(f"⚠️ Bus data fetcher: {e}")
        components_status['bus_fetcher'] = 'warning'
    
    try:
        from stt.stt_engine import stt_engine
        logger.info(f"✅ STT engine ready: {stt_engine.engine}")
        components_status['stt_engine'] = 'ready'
    except Exception as e:
        logger.warning(f"⚠️ STT engine: {e}")
        components_status['stt_engine'] = 'warning'
    
    try:
        from tts.tts_engine import tts_engine
        logger.info(f"✅ TTS engine ready: {tts_engine.engine}")
        components_status['tts_engine'] = 'ready'
    except Exception as e:
        logger.warning(f"⚠️ TTS engine: {e}")
        components_status['tts_engine'] = 'warning'
    
    try:
        from nlu.nlu_engine import nlu_engine
        logger.info(f"✅ NLU engine ready: {nlu_engine.engine}")
        components_status['nlu_engine'] = 'ready'
    except Exception as e:
        logger.warning(f"⚠️ NLU engine: {e}")
        components_status['nlu_engine'] = 'warning'
    
    # Test Gemini connection
    if settings.is_gemini_location_enabled():
        logger.info("✅ Gemini AI configured for location conversion")
        components_status['gemini_ai'] = 'ready'
    else:
        logger.warning("⚠️ Gemini API key not configured")
        components_status['gemini_ai'] = 'not_configured'
    
    # Log component status summary
    ready_count = sum(1 for status in components_status.values() if status == 'ready')
    total_count = len(components_status)
    logger.info(f"📊 Component Status: {ready_count}/{total_count} ready")
    
    logger.info("✅ AI Bus Assistant Backend API Ready")
    logger.info(f"🌐 Integration: https://bus-easebackend.onrender.com")
    
    yield
    
    # Shutdown
    logger.info("🛑 Backend API shutting down...")
    try:
        from fetcher.fetch_data import bus_data_fetcher
        await bus_data_fetcher.close_session()
        logger.info("✅ Bus data fetcher closed")
    except:
        pass

# Create FastAPI app (Backend only)
app = FastAPI(
    title="AI Bus Assistant Backend API",
    version="1.0.0",
    description="""
    **Backend API for AI Bus Assistant with Complete Voice Processing**
    
    ## 🚀 Features
    - 🎤 **Voice Processing**: Complete STT, NLU, TTS pipeline
    - 🤖 **Gemini AI Integration**: Smart location conversion & NLU
    - 🌍 **Multilingual Support**: English, Hindi, Punjabi with auto-detection
    - 🚌 **Real-time Bus Data**: Live integration with bus-easebackend.onrender.com
    - 🔊 **Audio Responses**: Natural TTS in multiple languages
    - 📱 **WebSocket Support**: Real-time communication
    
    ## 🌐 Integration
    - **Bus Data Source**: https://bus-easebackend.onrender.com
    - **Frontend**: Connect your frontend to this backend
    - **CORS**: Configured for frontend integration
    
    ## 🎯 Main Endpoints
    
    ### Voice Processing
    - `POST /api/v1/conversation/start` - Start conversation session
    - `POST /api/v1/voice/query` - Process text voice queries
    - `POST /api/v1/voice/process` - Process audio file uploads
    - `GET /api/v1/voice/languages` - Get supported languages
    
    ### Bus Data
    - `GET /api/v1/bus/search` - Search buses by route
    - `GET /api/v1/bus/by-name/{name}` - Get bus by name/number
    - `GET /api/v1/bus/{bus_id}` - Get bus details by ID
    
    ### System
    - `GET /health` - Health check
    - `GET /api/v1/status` - System status
    - `WebSocket /api/v1/ws` - Real-time communication
    
    ## 💡 Usage Example
    ```
    // Frontend Integration
    const API_BASE_URL = 'https://your-backend.onrender.com';
    
    const response = await fetch(`${API_BASE_URL}/api/v1/voice/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text: 'Where is bus 101A?',
        detect_language: true
      })
    });
    
    const result = await response.json();
    // result.response_text: "Bus 101A is at Central Station, Bangalore"
    // result.audio_url: "/static/audio/response_xyz.wav"
    // result.detected_language: "en"
    ```
    """,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
    debug=settings.debug
)

# CORS Configuration - Optimized for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*"  # For development - configure specific domains in production
        # "https://your-frontend-domain.com",
        # "https://your-app.onrender.com"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=[
        "Accept",
        "Accept-Language", 
        "Content-Language",
        "Content-Type",
        "Authorization",
        "X-Requested-With",
        "Origin",
        "Cache-Control",
        "Pragma"
    ],
    expose_headers=["*"]
)

# Include API routes
app.include_router(api_router, prefix="/api/v1", tags=["AI Bus Assistant Backend"])

# Include WebSocket routes (if websocket.py exists)
try:
    from api.websocket import router as ws_router
    app.include_router(ws_router, prefix="/api/v1", tags=["WebSocket"])
    logger.info("✅ WebSocket routes included")
except ImportError:
    logger.info("⚠️ WebSocket routes not found - skipping")

# Static files (for audio responses)
static_dir = "static"
if not os.path.exists(static_dir):
    os.makedirs(static_dir, exist_ok=True)

app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Root endpoint - Backend API information
@app.get("/", tags=["Root"])
async def root():
    """Backend API root endpoint with comprehensive information"""
    return {
        "service": "AI Bus Assistant Backend API",
        "version": "1.0.0",
        "status": "active",
        "timestamp": datetime.now().isoformat(),
        "description": "Complete voice-powered bus tracking backend API",
        
        # API Information
        "api_info": {
            "documentation": "/docs",
            "interactive_docs": "/redoc", 
            "health_check": "/health",
            "openapi_spec": "/openapi.json"
        },
        
        # Main Endpoints
        "endpoints": {
            "voice_processing": {
                "conversation_start": "POST /api/v1/conversation/start",
                "voice_query": "POST /api/v1/voice/query",
                "voice_process": "POST /api/v1/voice/process",
                "supported_languages": "GET /api/v1/voice/languages"
            },
            "bus_data": {
                "search_buses": "GET /api/v1/bus/search?start=...&end=...",
                "get_by_name": "GET /api/v1/bus/by-name/{name}",
                "get_by_id": "GET /api/v1/bus/{bus_id}"
            },
            "system": {
                "status": "GET /api/v1/status",
                "websocket": "WebSocket /api/v1/ws"
            }
        },
        
        # Integration Information
        "integration": {
            "bus_backend": {
                "url": "https://bus-easebackend.onrender.com",
                "endpoints_used": [
                    "GET /api/buses/search?start=...&end=...",
                    "GET /api/buses/{bus_id}",
                    "GET /api/buses/by-name/{name}"
                ]
            },
            "gemini_ai": {
                "enabled": settings.is_gemini_location_enabled(),
                "features": ["location_conversion", "natural_language_understanding"]
            }
        },
        
        # Features
        "features": {
            "voice_processing": {
                "speech_to_text": "Vosk + Edge TTS",
                "text_to_speech": "Edge TTS (multilingual)",
                "language_detection": "Auto-detect with Gemini AI"
            },
            "multilingual_support": {
                "supported_languages": ["English", "Hindi", "Punjabi"],
                "auto_detection": True,
                "voice_responses": True
            },
            "real_time_data": {
                "bus_tracking": True,
                "gps_conversion": "Coordinates → Human-readable locations",
                "caching": f"{settings.bus_data_cache_duration} seconds"
            }
        },
        
        # Frontend Integration Guide
        "frontend_integration": {
            "base_url": "Use this API URL as your backend",
            "cors_enabled": True,
            "audio_files": "Served at /static/audio/filename.wav",
            "websocket_support": True,
            "example_usage": {
                "javascript": """
const API_URL = 'https://your-backend.onrender.com';
const response = await fetch(`${API_URL}/api/v1/voice/query`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    text: 'Where is bus 101A?',
    detect_language: true
  })
});
const data = await response.json();
// Use data.response_text and data.audio_url
                """
            }
        }
    }

# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    """Comprehensive health check for all components"""
    try:
        components = {}
        overall_status = "healthy"
        
        # Check each component
        try:
            from utils.language_utils import language_utils
            components["language_utils"] = {
                "status": "healthy",
                "supported_languages": language_utils.supported_languages,
                "default_language": language_utils.default_language
            }
        except Exception as e:
            components["language_utils"] = {"status": "error", "error": str(e)}
            overall_status = "degraded"
        
        try:
            from utils.audio_utils import audio_processor
            components["audio_processor"] = {
                "status": "healthy",
                "details": audio_processor.get_status()
            }
        except Exception as e:
            components["audio_processor"] = {"status": "error", "error": str(e)}
        
        try:
            from fetcher.fetch_data import bus_data_fetcher
            components["bus_fetcher"] = {
                "status": "healthy",
                "backend_url": bus_data_fetcher.base_url,
                "cache_duration": bus_data_fetcher.cache_duration
            }
        except Exception as e:
            components["bus_fetcher"] = {"status": "error", "error": str(e)}
        
        components["gemini_ai"] = {
            "status": "configured" if settings.is_gemini_location_enabled() else "not_configured",
            "enabled": settings.is_gemini_location_enabled()
        }
        
        # Overall health
        error_count = sum(1 for comp in components.values() if comp.get("status") == "error")
        if error_count > 2:
            overall_status = "unhealthy"
        elif error_count > 0:
            overall_status = "degraded"
        
        return {
            "status": overall_status,
            "service": "AI Bus Assistant Backend API",
            "version": "1.0.0",
            "timestamp": datetime.now().isoformat(),
            "uptime_check": "healthy",
            "components": components,
            "api_endpoints": {
                "total_available": len(app.routes),
                "main_endpoints": [
                    "/api/v1/conversation/start",
                    "/api/v1/voice/query", 
                    "/api/v1/voice/process",
                    "/api/v1/bus/search",
                    "/api/v1/bus/by-name/{name}"
                ]
            },
            "integration_status": {
                "bus_backend": "connected",
                "static_files": "available",
                "cors": "configured"
            }
        }
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        raise HTTPException(
            status_code=503, 
            detail={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Custom 404 handler"""
    return {
        "error": "Endpoint not found",
        "message": f"The requested path '{request.url.path}' was not found",
        "status_code": 404,
        "available_endpoints": {
            "documentation": "/docs",
            "health_check": "/health",
            "voice_query": "/api/v1/voice/query",
            "conversation_start": "/api/v1/conversation/start",
            "bus_search": "/api/v1/bus/search"
        },
        "suggestion": "Check the API documentation at /docs"
    }

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Custom 500 handler"""
    logger.error(f"Internal server error: {exc}")
    return {
        "error": "Internal server error",
        "message": "An unexpected error occurred while processing your request",
        "status_code": 500,
        "timestamp": datetime.now().isoformat(),
        "support": "Check logs for more details"
    }

# Production server startup
if __name__ == "__main__":
    # Get port from environment (Render sets this automatically)
    port = int(os.environ.get("PORT", 10000))
    
    logger.info(f"🚀 Starting AI Bus Assistant Backend API")
    logger.info(f"🌐 Environment: {'Production' if not settings.debug else 'Development'}")
    logger.info(f"📡 Port: {port}")
    
    # Production configuration
    if settings.debug:
        # Development mode
        logger.info("🔧 Running in development mode with auto-reload")
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=port,
            reload=True,
            log_level="info"
        )
    else:
        # Production mode
        logger.info("🚀 Running in production mode")
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=port,
            workers=1,
            log_level="info",
            access_log=False  # Disable access logs in production for performance
        )
