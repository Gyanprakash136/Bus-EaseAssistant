"""
Audio Utils - Production Ready with Enhanced Processing & Validation
Python 3.13 Compatible with comprehensive audio handling
"""
import os
import uuid
import time
import wave
import tempfile
import struct
from typing import Dict, Any, Optional, Tuple, Union
import logging
from datetime import datetime
from config.settings import settings

logger = logging.getLogger(__name__)

class AudioProcessor:
    """Enhanced audio processor with production-grade features and validation."""
    
    def __init__(self):
        # Audio configuration
        self.chunk_size = getattr(settings, 'audio_chunk_size', getattr(settings, 'chunk_size', 1024))
        self.sample_rate = getattr(settings, 'audio_sample_rate', getattr(settings, 'sample_rate', 16000))
        self.channels = getattr(settings, 'audio_channels', getattr(settings, 'channels', 1))
        self.format = getattr(settings, 'audio_format', 'wav')
        self.temp_dir = getattr(settings, 'audio_temp_dir', 'temp/audio')
        self.max_audio_duration = getattr(settings, 'max_audio_duration', 30)
        
        # Audio validation settings
        self.min_audio_size = 100  # Minimum audio size in bytes
        self.max_audio_size = 50 * 1024 * 1024  # Maximum 50MB
        self.supported_formats = ['wav', 'raw', 'pcm']
        
        # File management
        self.max_temp_files = 100
        self.temp_file_lifetime = 3600  # 1 hour in seconds
        
        # Statistics
        self.stats = {
            'files_processed': 0,
            'files_saved': 0,
            'files_loaded': 0,
            'validation_failures': 0,
            'total_bytes_processed': 0,
            'average_file_size': 0,
            'last_operation_time': None
        }
        
        # Initialize
        self.initialize_directories()
        logger.info(f"✅ AudioProcessor initialized - Python 3.13 compatible")
        logger.info(f"🎵 Config: {self.sample_rate}Hz, {self.channels}ch, {self.format}")
    
    def initialize_directories(self):
        """Initialize and validate audio directories."""
        try:
            os.makedirs(self.temp_dir, exist_ok=True)
            
            # Test write permissions
            test_file = os.path.join(self.temp_dir, 'test_permissions.tmp')
            try:
                with open(test_file, 'wb') as f:
                    f.write(b'test')
                os.remove(test_file)
                logger.info(f"✅ Audio temp directory ready: {self.temp_dir}")
            except Exception as e:
                logger.error(f"❌ Audio temp directory not writable: {e}")
                # Use system temp as fallback
                self.temp_dir = tempfile.gettempdir()
                logger.warning(f"⚠️ Using system temp directory: {self.temp_dir}")
                
        except Exception as e:
            logger.error(f"❌ Audio directory initialization error: {e}")
            self.temp_dir = tempfile.gettempdir()
    
    def process_audio_data(self, audio_data: bytes, input_format: str = 'wav') -> Dict[str, Any]:
        """Enhanced audio data processing with validation and analysis."""
        try:
            # Update statistics
            self.stats['files_processed'] += 1
            self.stats['total_bytes_processed'] += len(audio_data)
            self.stats['last_operation_time'] = datetime.now().isoformat()
            
            # Update average file size
            self.stats['average_file_size'] = self.stats['total_bytes_processed'] / self.stats['files_processed']
            
            logger.info(f"🔄 Processing audio: {len(audio_data)} bytes, format: {input_format}")
            
            # Validate input
            validation_result = self.validate_audio_data(audio_data, input_format)
            if not validation_result['valid']:
                self.stats['validation_failures'] += 1
                return {
                    "success": False,
                    "error": validation_result['error'],
                    "validation": validation_result
                }
            
            # Analyze audio properties
            audio_info = self.analyze_audio_properties(audio_data, input_format)
            
            # Process based on format
            if input_format.lower() == 'wav':
                processed_result = self.process_wav_data(audio_data)
            else:
                processed_result = self.process_raw_data(audio_data)
            
            # Combine results
            result = {
                "success": True,
                "processed_data": processed_result['data'],
                "original_size_bytes": len(audio_data),
                "processed_size_bytes": len(processed_result['data']),
                "compression_ratio": len(processed_result['data']) / len(audio_data),
                "input_format": input_format,
                "output_format": self.format,
                **audio_info,
                **processed_result.get('properties', {}),
                "processing_time": time.time(),
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"✅ Audio processing completed: {result['duration_seconds']:.2f}s duration")
            return result
            
        except Exception as e:
            logger.error(f"❌ Audio processing error: {e}")
            self.stats['validation_failures'] += 1
            return {
                "success": False,
                "error": str(e),
                "original_size_bytes": len(audio_data) if audio_data else 0,
                "input_format": input_format,
                "timestamp": datetime.now().isoformat()
            }
    
    def validate_audio_data(self, audio_data: bytes, input_format: str) -> Dict[str, Any]:
        """Comprehensive audio data validation."""
        try:
            # Check if data exists
            if not audio_data:
                return {"valid": False, "error": "Audio data is empty"}
            
            # Check size limits
            if len(audio_data) < self.min_audio_size:
                return {"valid": False, "error": f"Audio too small: {len(audio_data)} bytes (min: {self.min_audio_size})"}
            
            if len(audio_data) > self.max_audio_size:
                return {"valid": False, "error": f"Audio too large: {len(audio_data)} bytes (max: {self.max_audio_size})"}
            
            # Check format support
            if input_format.lower() not in self.supported_formats:
                return {"valid": False, "error": f"Unsupported format: {input_format} (supported: {self.supported_formats})"}
            
            # Format-specific validation
            if input_format.lower() == 'wav':
                wav_validation = self.validate_wav_format(audio_data)
                if not wav_validation['valid']:
                    return wav_validation
            
            # Check for silence or noise
            activity_check = self.detect_voice_activity(audio_data)
            
            return {
                "valid": True,
                "size_bytes": len(audio_data),
                "format": input_format,
                "voice_activity_detected": activity_check,
                "validation_time": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"valid": False, "error": f"Validation error: {str(e)}"}
    
    def validate_wav_format(self, wav_data: bytes) -> Dict[str, Any]:
        """Validate WAV file format."""
        try:
            if len(wav_data) < 44:  # Minimum WAV header size
                return {"valid": False, "error": "WAV file too small for valid header"}
            
            # Check RIFF header
            if not wav_data.startswith(b'RIFF'):
                return {"valid": False, "error": "Invalid WAV format: missing RIFF header"}
            
            # Check WAVE identifier
            if wav_data[8:12] != b'WAVE':
                return {"valid": False, "error": "Invalid WAV format: missing WAVE identifier"}
            
            # Check fmt chunk
            if wav_data[12:16] != b'fmt ':
                return {"valid": False, "error": "Invalid WAV format: missing fmt chunk"}
            
            # Extract format information
            try:
                fmt_chunk_size = struct.unpack('<I', wav_data[16:20])[0]
                audio_format = struct.unpack('<H', wav_data[20:22])[0]
                num_channels = struct.unpack('<H', wav_data[22:24])[0]
                sample_rate = struct.unpack('<I', wav_data[24:28])[0]
                
                # Validate reasonable values
                if audio_format not in [1, 3]:  # PCM or Float
                    return {"valid": False, "error": f"Unsupported audio format: {audio_format}"}
                
                if num_channels < 1 or num_channels > 8:
                    return {"valid": False, "error": f"Invalid channel count: {num_channels}"}
                
                if sample_rate < 8000 or sample_rate > 192000:
                    return {"valid": False, "error": f"Invalid sample rate: {sample_rate}"}
                
                return {
                    "valid": True,
                    "wav_properties": {
                        "audio_format": audio_format,
                        "channels": num_channels,
                        "sample_rate": sample_rate,
                        "fmt_chunk_size": fmt_chunk_size
                    }
                }
                
            except struct.error as e:
                return {"valid": False, "error": f"WAV header parsing error: {str(e)}"}
                
        except Exception as e:
            return {"valid": False, "error": f"WAV validation error: {str(e)}"}
    
    def analyze_audio_properties(self, audio_data: bytes, input_format: str) -> Dict[str, Any]:
        """Analyze audio properties and characteristics."""
        try:
            properties = {
                "size_bytes": len(audio_data),
                "format": input_format
            }
            
            if input_format.lower() == 'wav':
                # Parse WAV header for detailed info
                if len(audio_data) >= 44:
                    try:
                        # Extract WAV properties
                        num_channels = struct.unpack('<H', audio_data[22:24])[0]
                        sample_rate = struct.unpack('<I', audio_data[24:28])[0]
                        byte_rate = struct.unpack('<I', audio_data[28:32])[0]
                        bits_per_sample = struct.unpack('<H', audio_data[34:36])[0]
                        
                        # Calculate duration
                        data_size = len(audio_data) - 44  # Assume standard 44-byte header
                        duration_seconds = data_size / byte_rate if byte_rate > 0 else 0
                        
                        properties.update({
                            "sample_rate": sample_rate,
                            "channels": num_channels,
                            "bits_per_sample": bits_per_sample,
                            "byte_rate": byte_rate,
                            "duration_seconds": round(duration_seconds, 3),
                            "data_size_bytes": data_size
                        })
                        
                    except struct.error:
                        # Fallback estimation
                        properties.update({
                            "sample_rate": self.sample_rate,
                            "channels": self.channels,
                            "duration_seconds": round((len(audio_data) - 44) / (self.sample_rate * 2 * self.channels), 3)
                        })
            else:
                # Raw audio estimation
                bytes_per_sample = 2  # Assume 16-bit
                total_samples = len(audio_data) // (bytes_per_sample * self.channels)
                duration_seconds = total_samples / self.sample_rate
                
                properties.update({
                    "sample_rate": self.sample_rate,
                    "channels": self.channels,
                    "duration_seconds": round(duration_seconds, 3),
                    "estimated": True
                })
            
            # Check duration limits
            if properties.get('duration_seconds', 0) > self.max_audio_duration:
                properties['warning'] = f"Audio duration exceeds limit: {properties['duration_seconds']}s > {self.max_audio_duration}s"
            
            return properties
            
        except Exception as e:
            logger.warning(f"⚠️ Audio analysis error: {e}")
            return {
                "size_bytes": len(audio_data),
                "format": input_format,
                "analysis_error": str(e)
            }
    
    def process_wav_data(self, wav_data: bytes) -> Dict[str, Any]:
        """Process WAV format audio data."""
        try:
            # For now, return as-is (could add format conversion here)
            return {
                "data": wav_data,
                "properties": {
                    "processing_type": "wav_passthrough",
                    "header_preserved": True
                }
            }
        except Exception as e:
            logger.error(f"❌ WAV processing error: {e}")
            raise
    
    def process_raw_data(self, raw_data: bytes) -> Dict[str, Any]:
        """Process raw audio data."""
        try:
            # Convert raw data to WAV format
            wav_data = self.raw_to_wav(raw_data)
            
            return {
                "data": wav_data,
                "properties": {
                    "processing_type": "raw_to_wav_conversion",
                    "header_added": True
                }
            }
        except Exception as e:
            logger.error(f"❌ Raw audio processing error: {e}")
            return {
                "data": raw_data,
                "properties": {
                    "processing_type": "raw_passthrough",
                    "conversion_failed": str(e)
                }
            }
    
    def raw_to_wav(self, raw_data: bytes) -> bytes:
        """Convert raw audio data to WAV format."""
        try:
            # Create WAV header
            data_size = len(raw_data)
            file_size = data_size + 36
            
            # WAV header components
            riff_header = b'RIFF'
            file_size_bytes = struct.pack('<I', file_size)
            wave_header = b'WAVE'
            fmt_header = b'fmt '
            fmt_chunk_size = struct.pack('<I', 16)
            audio_format = struct.pack('<H', 1)  # PCM
            num_channels = struct.pack('<H', self.channels)
            sample_rate_bytes = struct.pack('<I', self.sample_rate)
            byte_rate = struct.pack('<I', self.sample_rate * self.channels * 2)  # Assume 16-bit
            block_align = struct.pack('<H', self.channels * 2)
            bits_per_sample = struct.pack('<H', 16)
            data_header = b'data'
            data_size_bytes = struct.pack('<I', data_size)
            
            # Combine header with data
            wav_header = (riff_header + file_size_bytes + wave_header + 
                         fmt_header + fmt_chunk_size + audio_format + 
                         num_channels + sample_rate_bytes + byte_rate + 
                         block_align + bits_per_sample + data_header + 
                         data_size_bytes)
            
            return wav_header + raw_data
            
        except Exception as e:
            logger.error(f"❌ Raw to WAV conversion error: {e}")
            raise
    
    def detect_voice_activity(self, audio_data: bytes, threshold: float = 0.01) -> bool:
        """Enhanced voice activity detection."""
        try:
            # Simple energy-based VAD
            if len(audio_data) < self.min_audio_size:
                return False
            
            # Skip WAV header if present
            data_start = 44 if audio_data.startswith(b'RIFF') else 0
            audio_samples = audio_data[data_start:]
            
            if len(audio_samples) == 0:
                return False
            
            # Calculate energy (simplified)
            energy = 0
            sample_count = 0
            
            # Process as 16-bit samples
            for i in range(0, len(audio_samples) - 1, 2):
                try:
                    sample = struct.unpack('<h', audio_samples[i:i+2])[0]
                    energy += abs(sample)
                    sample_count += 1
                except struct.error:
                    continue
            
            if sample_count == 0:
                return False
            
            # Normalize energy
            avg_energy = energy / sample_count
            normalized_energy = avg_energy / 32768.0  # Normalize for 16-bit
            
            activity_detected = normalized_energy > threshold
            
            logger.debug(f"🎵 Voice activity: {activity_detected} (energy: {normalized_energy:.4f}, threshold: {threshold})")
            
            return activity_detected
            
        except Exception as e:
            logger.warning(f"⚠️ Voice activity detection error: {e}")
            # Fallback: assume activity if audio is large enough
            return len(audio_data) > 1000
    
    def save_audio_file(self, audio_data: bytes, filename: str = None) -> str:
        """Enhanced audio file saving with validation and cleanup."""
        try:
            self.stats['files_saved'] += 1
            self.stats['last_operation_time'] = datetime.now().isoformat()
            
            # Generate filename if not provided
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                filename = f"audio_{timestamp}.{self.format}"
            
            # Ensure proper extension
            if not filename.endswith(f".{self.format}"):
                name_without_ext = os.path.splitext(filename)[0]
                filename = f"{name_without_ext}.{self.format}"
            
            filepath = os.path.join(self.temp_dir, filename)
            
            # Validate audio data before saving
            validation = self.validate_audio_data(audio_data, self.format)
            if not validation['valid']:
                logger.warning(f"⚠️ Saving potentially invalid audio: {validation['error']}")
            
            # Save file
            with open(filepath, 'wb') as f:
                f.write(audio_data)
            
            logger.info(f"💾 Audio saved: {filename} ({len(audio_data)} bytes)")
            
            # Cleanup old files periodically
            if self.stats['files_saved'] % 10 == 0:
                self.cleanup_temp_files()
            
            return filepath
            
        except Exception as e:
            logger.error(f"❌ Audio save error: {e}")
            raise
    
    def load_audio_file(self, filepath: str) -> bytes:
        """Enhanced audio file loading with validation."""
        try:
            self.stats['files_loaded'] += 1
            self.stats['last_operation_time'] = datetime.now().isoformat()
            
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Audio file not found: {filepath}")
            
            file_size = os.path.getsize(filepath)
            if file_size == 0:
                raise ValueError(f"Audio file is empty: {filepath}")
            
            if file_size > self.max_audio_size:
                raise ValueError(f"Audio file too large: {file_size} bytes (max: {self.max_audio_size})")
            
            with open(filepath, 'rb') as f:
                audio_data = f.read()
            
            logger.info(f"📂 Audio loaded: {os.path.basename(filepath)} ({len(audio_data)} bytes)")
            
            return audio_data
            
        except Exception as e:
            logger.error(f"❌ Audio load error: {e}")
            raise
    
    def cleanup_temp_files(self):
        """Clean up old temporary audio files."""
        try:
            current_time = time.time()
            cleaned_count = 0
            
            for filename in os.listdir(self.temp_dir):
                filepath = os.path.join(self.temp_dir, filename)
                
                if os.path.isfile(filepath):
                    file_age = current_time - os.path.getmtime(filepath)
                    
                    if file_age > self.temp_file_lifetime:
                        try:
                            os.remove(filepath)
                            cleaned_count += 1
                            logger.debug(f"🗑️ Cleaned old audio file: {filename}")
                        except Exception as e:
                            logger.warning(f"⚠️ Failed to clean file {filename}: {e}")
            
            # Also limit total number of files
            files = [f for f in os.listdir(self.temp_dir) if os.path.isfile(os.path.join(self.temp_dir, f))]
            if len(files) > self.max_temp_files:
                # Remove oldest files
                files_with_time = [(f, os.path.getmtime(os.path.join(self.temp_dir, f))) for f in files]
                files_with_time.sort(key=lambda x: x[1])  # Sort by modification time
                
                files_to_remove = files_with_time[:len(files_with_time) - self.max_temp_files]
                for filename, _ in files_to_remove:
                    try:
                        os.remove(os.path.join(self.temp_dir, filename))
                        cleaned_count += 1
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to remove excess file {filename}: {e}")
            
            if cleaned_count > 0:
                logger.info(f"🧹 Cleaned {cleaned_count} temporary audio files")
                
        except Exception as e:
            logger.error(f"❌ Temp file cleanup error: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive audio processor status."""
        try:
            # Calculate directory info
            temp_files = len([f for f in os.listdir(self.temp_dir) if os.path.isfile(os.path.join(self.temp_dir, f))])
            temp_size = sum(
                os.path.getsize(os.path.join(self.temp_dir, f))
                for f in os.listdir(self.temp_dir)
                if os.path.isfile(os.path.join(self.temp_dir, f))
            )
            
            return {
                "initialized": True,
                "configuration": {
                    "sample_rate": self.sample_rate,
                    "channels": self.channels,
                    "chunk_size": self.chunk_size,
                    "format": self.format,
                    "temp_dir": self.temp_dir,
                    "max_audio_duration": self.max_audio_duration
                },
                "limits": {
                    "min_audio_size": self.min_audio_size,
                    "max_audio_size": self.max_audio_size,
                    "max_temp_files": self.max_temp_files,
                    "temp_file_lifetime_hours": self.temp_file_lifetime / 3600
                },
                "supported_formats": self.supported_formats,
                "statistics": self.stats,
                "temp_directory_info": {
                    "path": self.temp_dir,
                    "file_count": temp_files,
                    "total_size_mb": round(temp_size / 1024 / 1024, 2),
                    "writable": os.access(self.temp_dir, os.W_OK)
                },
                "performance": {
                    "success_rate": (
                        (self.stats['files_processed'] - self.stats['validation_failures']) / 
                        max(self.stats['files_processed'], 1) * 100
                        if self.stats['files_processed'] > 0 else 100
                    ),
                    "average_file_size_kb": round(self.stats['average_file_size'] / 1024, 2)
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Status generation error: {e}")
            return {
                "initialized": True,
                "error": str(e),
                "basic_config": {
                    "sample_rate": self.sample_rate,
                    "channels": self.channels,
                    "format": self.format
                }
            }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform audio processor health check."""
        try:
            # Test basic functionality
            test_data = b'\x00\x01' * 1000  # 2000 bytes of test data
            
            start_time = time.time()
            
            # Test processing
            process_result = self.process_audio_data(test_data, 'raw')
            
            # Test file operations
            temp_file = self.save_audio_file(test_data)
            loaded_data = self.load_audio_file(temp_file)
            
            # Cleanup test file
            if os.path.exists(temp_file):
                os.remove(temp_file)
            
            end_time = time.time()
            
            # Verify operations
            processing_success = process_result.get('success', False)
            file_ops_success = len(loaded_data) == len(test_data)
            
            return {
                "status": "healthy" if processing_success and file_ops_success else "degraded",
                "processing_works": processing_success,
                "file_operations_work": file_ops_success,
                "temp_directory_writable": os.access(self.temp_dir, os.W_OK),
                "response_time_seconds": round(end_time - start_time, 3),
                "last_check": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "last_check": datetime.now().isoformat()
            }
    
    def reset_statistics(self):
        """Reset audio processor statistics."""
        self.stats = {
            'files_processed': 0,
            'files_saved': 0,
            'files_loaded': 0,
            'validation_failures': 0,
            'total_bytes_processed': 0,
            'average_file_size': 0,
            'last_operation_time': None
        }
        logger.info("📊 Audio processor statistics reset")

# Global instance
audio_processor = AudioProcessor()
