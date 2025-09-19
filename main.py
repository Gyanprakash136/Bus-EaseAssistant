"""
AI Bus Assistant - Production Main (Render Ready)
"""
import uvicorn
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
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
    logger.info("🚀 AI Bus Assistant API Starting (Production)")
    logger.info(f"Environment: {'Production' if not settings.debug else 'Development'}")
    logger.info(f"Host: {settings.host}:{settings.port}")
    
    # Create directories
    directories = ["static/audio", settings.audio_temp_dir, "temp/cache", "logs"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    # Initialize components
    try:
        from utils.audio_utils import audio_processor
        logger.info("✅ Audio processor ready")
    except Exception as e:
        logger.error(f"❌ Audio processor error: {e}")
    
    try:
        from fetcher.fetch_data import bus_data_fetcher
        await bus_data_fetcher.initialize_session()
        logger.info("✅ Bus data fetcher ready")
    except Exception as e:
        logger.error(f"❌ Bus data fetcher error: {e}")
    
    logger.info("✅ AI Bus Assistant API Ready (Production)")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down...")
    try:
        from fetcher.fetch_data import bus_data_fetcher
        await bus_data_fetcher.close_session()
    except:
        pass

# Create FastAPI app
app = FastAPI(
    title="AI Bus Assistant API",
    version="1.0.0",
    description="Production AI Bus Assistant with Voice Interface",
    docs_url="/docs" if settings.debug else None,  # Disable docs in production
    redoc_url="/redoc" if settings.debug else None,
    lifespan=lifespan
)

# CORS for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly for production
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# API routes
app.include_router(api_router, prefix="/api/v1", tags=["AI Bus Assistant"])

# Static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Root endpoint
@app.get("/")
async def root():
    return FileResponse('static/index.html')

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "AI Bus Assistant API",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "environment": "production" if not settings.debug else "development"
    }

# Production server
if __name__ == "__main__":
    # Get port from environment (Render sets this)
    port = int(os.environ.get("PORT", settings.port))
    
    if settings.debug:
        # Development
        uvicorn.run(
            "main:app",
            host=settings.host,
            port=port,
            reload=True
        )
    else:
        # Production
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=port,
            workers=1
        )
