"""
Audio Utils - Python 3.13 Compatible
"""
import os
import uuid
import time
from typing import Dict, Any
from config.settings import settings

class AudioProcessor:
    """Minimal audio processor without external dependencies."""
    
    def __init__(self):
        self.chunk_size = getattr(settings, 'audio_chunk_size', 1024)
        self.sample_rate = getattr(settings, 'audio_sample_rate', 16000)
        self.channels = getattr(settings, 'audio_channels', 1)
        self.format = getattr(settings, 'audio_format', 'wav')
        self.temp_dir = getattr(settings, 'audio_temp_dir', 'temp/audio')
        
        os.makedirs(self.temp_dir, exist_ok=True)
        print(f"AudioProcessor initialized - Python 3.13 compatible")
    
    def process_audio_data(self, audio_data: bytes, input_format: str = 'wav') -> Dict[str, Any]:
        return {
            "success": True,
            "processed_data": audio_data,
            "duration": len(audio_data) / (self.sample_rate * 2),  # Rough estimate
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "format": self.format,
            "size_bytes": len(audio_data)
        }
    
    def detect_voice_activity(self, audio_data: bytes, sample_rate: int = None) -> bool:
        return len(audio_data) > 1000
    
    def save_audio_file(self, audio_data: bytes, filename: str = None) -> str:
        if not filename:
            filename = f"audio_{uuid.uuid4().hex[:8]}.{self.format}"
        
        filepath = os.path.join(self.temp_dir, filename)
        with open(filepath, 'wb') as f:
            f.write(audio_data)
        
        return filepath
    
    def load_audio_file(self, filepath: str) -> bytes:
        with open(filepath, 'rb') as f:
            return f.read()
    
    def get_status(self) -> Dict[str, Any]:
        return {
            "initialized": True,
            "sample_rate": self.sample_rate,
            "chunk_size": self.chunk_size,
            "channels": self.channels,
            "format": self.format,
            "temp_dir": self.temp_dir
        }

audio_processor = AudioProcessor()
