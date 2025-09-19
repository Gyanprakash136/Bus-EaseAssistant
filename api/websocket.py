"""
WebSocket Support for Real-time Communication - Production Ready
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import json
import logging
from typing import Dict, Any
from datetime import datetime

# Import engines with error handling
try:
    from nlu.nlu_engine import nlu_engine
except ImportError:
    nlu_engine = None

try:
    from tts.tts_engine import tts_engine  
except ImportError:
    tts_engine = None

try:
    from utils.language_utils import language_utils
except ImportError:
    language_utils = None

logger = logging.getLogger(__name__)
router = APIRouter()

class ConnectionManager:
    """Enhanced WebSocket connection manager."""
    
    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self.connection_info: dict = {}
    
    async def connect(self, websocket: WebSocket, client_id: str = None):
        await websocket.accept()
        self.active_connections.append(websocket)
        
        # Store connection info
        self.connection_info[websocket] = {
            "client_id": client_id or f"client_{len(self.active_connections)}",
            "connected_at": datetime.now().isoformat(),
            "message_count": 0
        }
        
        logger.info(f"🔌 WebSocket connected. Total: {len(self.active_connections)}, Client: {client_id}")
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            client_info = self.connection_info.pop(websocket, {})
            client_id = client_info.get("client_id", "unknown")
            logger.info(f"🔌 WebSocket disconnected. Total: {len(self.active_connections)}, Client: {client_id}")
    
    async def send_personal_message(self, message: dict, websocket: WebSocket):
        try:
            # Add timestamp to all messages
            message["timestamp"] = datetime.now().isoformat()
            await websocket.send_text(json.dumps(message))
            
            # Update message count
            if websocket in self.connection_info:
                self.connection_info[websocket]["message_count"] += 1
                
        except Exception as e:
            logger.error(f"Failed to send WebSocket message: {e}")
    
    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients."""
        message["timestamp"] = datetime.now().isoformat()
        disconnected = []
        
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except Exception as e:
                logger.warning(f"Failed to broadcast to connection: {e}")
                disconnected.append(connection)
        
        # Clean up disconnected clients
        for connection in disconnected:
            self.disconnect(connection)
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        return {
            "total_connections": len(self.active_connections),
            "connections": [
                {
                    "client_id": info["client_id"],
                    "connected_at": info["connected_at"], 
                    "message_count": info["message_count"]
                }
                for info in self.connection_info.values()
            ]
        }

manager = ConnectionManager()

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Enhanced WebSocket endpoint for real-time communication."""
    client_id = None
    
    try:
        # Extract client ID from query params if available
        client_id = websocket.query_params.get("client_id")
        await manager.connect(websocket, client_id)
        
        # Send welcome message with system status
        welcome_data = {
            "type": "welcome",
            "message": "Connected to AI Bus Assistant",
            "client_id": manager.connection_info[websocket]["client_id"],
            "supported_languages": language_utils.supported_languages if language_utils else ["en", "hi", "pa"],
            "features": {
                "voice_processing": nlu_engine is not None and tts_engine is not None,
                "language_detection": language_utils is not None,
                "real_time_bus_data": True
            },
            "available_message_types": [
                "ping", "voice_query", "bus_search", "language_detect", "status_request"
            ]
        }
        
        await manager.send_personal_message(welcome_data, websocket)
        
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                message_type = message.get('type')
                
                # Handle different message types
                if message_type == 'ping':
                    await handle_ping(websocket, message)
                    
                elif message_type == 'voice_query':
                    await handle_voice_query(websocket, message)
                    
                elif message_type == 'bus_search':
                    await handle_bus_search(websocket, message)
                    
                elif message_type == 'language_detect':
                    await handle_language_detect(websocket, message)
                    
                elif message_type == 'status_request':
                    await handle_status_request(websocket, message)
                    
                else:
                    await manager.send_personal_message({
                        "type": "error",
                        "message": f"Unknown message type: {message_type}",
                        "available_types": ["ping", "voice_query", "bus_search", "language_detect", "status_request"]
                    }, websocket)
                    
            except json.JSONDecodeError:
                await manager.send_personal_message({
                    "type": "error",
                    "message": "Invalid JSON format",
                    "example": {
                        "type": "voice_query",
                        "text": "Where is bus 101A?",
                        "language": "en"
                    }
                }, websocket)
                
            except Exception as e:
                logger.error(f"WebSocket message processing error: {e}")
                await manager.send_personal_message({
                    "type": "error",
                    "message": "Internal processing error",
                    "error_details": str(e)
                }, websocket)
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info(f"WebSocket client {client_id} disconnected normally")
        
    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {e}")
        manager.disconnect(websocket)

async def handle_ping(websocket: WebSocket, message: dict):
    """Handle ping messages for connection testing."""
    await manager.send_personal_message({
        "type": "pong",
        "original_timestamp": message.get('timestamp'),
        "server_time": datetime.now().isoformat()
    }, websocket)

async def handle_voice_query(websocket: WebSocket, message: dict):
    """Handle voice query messages."""
    try:
        text = message.get('text', '')
        language = message.get('language', 'en')
        detect_language = message.get('detect_language', True)
        
        if not text.strip():
            await manager.send_personal_message({
                "type": "error",
                "message": "Empty text provided"
            }, websocket)
            return
        
        # Language detection if enabled
        detected_language = language
        if detect_language and language_utils:
            detected_language = language_utils.detect_language(text)
            logger.info(f"🌍 WebSocket language detected: {text[:30]}... → {detected_language}")
        
        # Check NLU availability
        if not nlu_engine:
            await manager.send_personal_message({
                "type": "error",
                "message": "Voice processing not available - NLU engine not loaded"
            }, websocket)
            return
        
        # Process query
        nlu_result = await nlu_engine.process_query(text, detected_language)
        
        if nlu_result.get('success'):
            response_text = nlu_result['response']
            
            # Generate audio if TTS available
            audio_url = None
            if tts_engine:
                try:
                    tts_result = await tts_engine.text_to_speech(response_text, detected_language)
                    audio_url = tts_result.get('audio_url')
                except Exception as e:
                    logger.warning(f"WebSocket TTS failed: {e}")
            
            await manager.send_personal_message({
                "type": "voice_response",
                "query": text,
                "response": response_text,
                "detected_language": detected_language,
                "language_info": language_utils.get_language_info(detected_language) if language_utils else None,
                "audio_url": audio_url,
                "intent": nlu_result.get('intent'),
                "data": nlu_result.get('data'),
                "processing_successful": True
            }, websocket)
        else:
            await manager.send_personal_message({
                "type": "error",
                "message": "Failed to process voice query",
                "details": nlu_result.get('error', 'Unknown NLU error')
            }, websocket)
            
    except Exception as e:
        logger.error(f"Voice query handling error: {e}")
        await manager.send_personal_message({
            "type": "error",
            "message": "Voice query processing failed",
            "error_details": str(e)
        }, websocket)

async def handle_bus_search(websocket: WebSocket, message: dict):
    """Handle bus search requests."""
    try:
        start = message.get('start')
        end = message.get('end')
        bus_name = message.get('bus_name')
        
        # Import bus fetcher here to avoid circular imports
        try:
            from fetcher.fetch_data import bus_data_fetcher
            
            if bus_name:
                result = await bus_data_fetcher.get_bus_by_name(bus_name)
                response_type = "bus_details"
            else:
                result = await bus_data_fetcher.search_buses(start, end)
                response_type = "bus_search_results"
            
            await manager.send_personal_message({
                "type": response_type,
                "query": {
                    "start": start,
                    "end": end,
                    "bus_name": bus_name
                },
                "result": result
            }, websocket)
            
        except ImportError:
            await manager.send_personal_message({
                "type": "error",
                "message": "Bus data service not available"
            }, websocket)
            
    except Exception as e:
        logger.error(f"Bus search handling error: {e}")
        await manager.send_personal_message({
            "type": "error",
            "message": "Bus search failed",
            "error_details": str(e)
        }, websocket)

async def handle_language_detect(websocket: WebSocket, message: dict):
    """Handle language detection requests."""
    try:
        text = message.get('text', '')
        
        if not text.strip():
            await manager.send_personal_message({
                "type": "error",
                "message": "No text provided for language detection"
            }, websocket)
            return
        
        if language_utils:
            detected_language = language_utils.detect_language(text)
            language_info = language_utils.get_language_info(detected_language)
            
            await manager.send_personal_message({
                "type": "language_detected",
                "text": text,
                "detected_language": detected_language,
                "language_info": language_info,
                "confidence": "high" if len(text) > 20 else "medium"
            }, websocket)
        else:
            await manager.send_personal_message({
                "type": "error",
                "message": "Language detection not available"
            }, websocket)
            
    except Exception as e:
        logger.error(f"Language detection error: {e}")
        await manager.send_personal_message({
            "type": "error",
            "message": "Language detection failed",
            "error_details": str(e)
        }, websocket)

async def handle_status_request(websocket: WebSocket, message: dict):
    """Handle system status requests."""
    try:
        system_status = {
            "type": "system_status",
            "service": "AI Bus Assistant WebSocket",
            "status": "active",
            "components": {
                "nlu_engine": "available" if nlu_engine else "not_available",
                "tts_engine": "available" if tts_engine else "not_available", 
                "language_utils": "available" if language_utils else "not_available"
            },
            "connection_stats": manager.get_connection_stats(),
            "server_info": {
                "server_time": datetime.now().isoformat(),
                "uptime": "active"
            }
        }
        
        await manager.send_personal_message(system_status, websocket)
        
    except Exception as e:
        logger.error(f"Status request error: {e}")
        await manager.send_personal_message({
            "type": "error", 
            "message": "Failed to get system status",
            "error_details": str(e)
        }, websocket)

# Additional endpoint to get WebSocket stats via HTTP
@router.get("/ws/stats")
async def get_websocket_stats() -> Dict[str, Any]:
    """Get WebSocket connection statistics via HTTP."""
    return {
        "websocket_stats": manager.get_connection_stats(),
        "service_status": {
            "nlu_available": nlu_engine is not None,
            "tts_available": tts_engine is not None,
            "language_utils_available": language_utils is not None
        },
        "timestamp": datetime.now().isoformat()
    }
