"""
AI Bus Assistant Settings - Complete & Bug-Free
"""
import os
from typing import Optional, Literal, List
from pydantic import Field

try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings

class Settings(BaseSettings):
    """Complete application settings."""
    
    # Core Settings
    app_name: str = "AI Bus Assistant"
    app_version: str = "1.2.0"
    debug: bool = Field(default=False)
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    
    # API Keys
    gemini_api_key: str = Field(default="your_actual_gemini_api_key_here")
    gemini_model: str = Field(default="gemini-1.5-flash")
    
    # Audio Settings
    audio_temp_dir: str = Field(default="temp/audio")
    max_audio_duration: int = Field(default=30)
    audio_sample_rate: int = Field(default=16000)
    audio_chunk_size: int = Field(default=1024)
    audio_channels: int = Field(default=1)
    audio_format: str = Field(default="wav")
    
    # Legacy compatibility
    sample_rate: int = Field(default=16000)
    chunk_size: int = Field(default=1024)
    channels: int = Field(default=1)
    model_path: str = Field(default="./models")
    
    # TTS Settings
    tts_engine: str = Field(default="edge_tts")
    tts_voice_en: str = Field(default="en-US-AriaNeural")
    tts_voice_hi: str = Field(default="hi-IN-SwaraNeural")
    tts_voice_pa: str = Field(default="pa-IN-GulNeural")
    
    # STT Settings
    stt_engine: str = Field(default="vosk")
    vosk_model_path: str = Field(default="./models")
    enable_vosk: bool = Field(default=True)
    
    # Language Settings
    supported_languages: List[str] = Field(default=["en", "hi", "pa"])
    default_language: str = Field(default="en")
    enable_language_detection: bool = Field(default=True)
    
    # NLU Settings
    nlu_engine: str = Field(default="gemini")
    nlu_timeout: int = Field(default=10)
    
    # Bus API Settings - Updated with your backend
    bus_api_base_url: str = Field(default="https://bus-easebackend.onrender.com")
    bus_api_key: Optional[str] = Field(default=None)
    bus_data_cache_duration: int = Field(default=30)
    bus_data_timeout: int = Field(default=10)
    
    # Gemini Location Settings
    gemini_location_cache_duration: int = Field(default=3600)
    gemini_location_timeout: int = Field(default=5)
    enable_gemini_locations: bool = Field(default=True)
    fallback_to_manual_locations: bool = Field(default=True)
    location_detail_level: Literal["basic", "detailed", "verbose"] = Field(default="detailed")
    include_city_name: bool = Field(default=True)
    prefer_landmarks: bool = Field(default=True)
    
    # Advanced Gemini Settings
    gemini_location_model: str = Field(default="gemini-1.5-flash")
    gemini_location_temperature: float = Field(default=0.3)
    gemini_location_max_tokens: int = Field(default=100)
    
    # Regional Settings
    primary_region: str = Field(default="India")
    primary_cities: List[str] = Field(default=["Bangalore", "Mumbai", "Delhi", "Chennai", "Hyderabad", "Bhubaneswar"])
    location_context_keywords: List[str] = Field(default=[
        "station", "terminal", "airport", "railway", "metro", "bus stop",
        "junction", "circle", "cross", "road", "street", "market", "mall"
    ])
    
    # System Settings
    enable_websocket: bool = Field(default=True)
    cors_origins: List[str] = Field(default=["*"])
    cors_allow_credentials: bool = Field(default=True)
    log_level: str = Field(default="INFO")
    log_file: Optional[str] = Field(default=None)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"
    
    def is_gemini_location_enabled(self) -> bool:
        return (
            self.enable_gemini_locations and
            self.gemini_api_key and 
            self.gemini_api_key != "your_actual_gemini_api_key_here"
        )
    
    def get_location_prompt_template(self) -> str:
        return f"""
Convert GPS coordinates to location name for {self.primary_region}:

Instructions:
1. Provide a detailed location name with landmarks
2. Include the city name
3. Keep response under 60 characters
4. Focus on public transportation relevant locations

Respond with ONLY the location name, nothing else.
"""

def load_settings():
    try:
        return Settings()
    except Exception as e:
        print(f"Settings error: {e}")
        return Settings()

settings = load_settings()
