"""
WebSocket Support for Real-time Communication
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import json
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

router = APIRouter()

class ConnectionManager:
    """Manage WebSocket connections."""
    
    def __init__(self):
        self.active_connections: list[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: dict, websocket: WebSocket):
        await websocket.send_text(json.dumps(message))
    
    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except:
                # Connection might be closed
                pass

manager = ConnectionManager()

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time communication."""
    await manager.connect(websocket)
    
    try:
        # Send welcome message
        await manager.send_personal_message({
            "type": "welcome",
            "message": "Connected to AI Bus Assistant",
            "supported_languages": ["en", "hi", "pa"]
        }, websocket)
        
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                message_type = message.get('type')
                
                if message_type == 'ping':
                    await manager.send_personal_message({
                        "type": "pong",
                        "timestamp": message.get('timestamp')
                    }, websocket)
                
                elif message_type == 'voice_query':
                    # Process voice query via WebSocket
                    text = message.get('text', '')
                    language = message.get('language', 'en')
                    
                    # Import here to avoid circular imports
                    from nlu.nlu_engine import nlu_engine
                    from tts.tts_engine import tts_engine
                    
                    # Process query
                    nlu_result = await nlu_engine.process_query(text, language)
                    
                    if nlu_result.get('success'):
                        response_text = nlu_result['response']
                        
                        # Generate audio
                        tts_result = await tts_engine.text_to_speech(response_text, language)
                        
                        await manager.send_personal_message({
                            "type": "voice_response",
                            "query": text,
                            "response": response_text,
                            "audio_url": tts_result.get('audio_url'),
                            "intent": nlu_result.get('intent'),
                            "data": nlu_result.get('data')
                        }, websocket)
                    else:
                        await manager.send_personal_message({
                            "type": "error",
                            "message": "Failed to process query"
                        }, websocket)
                
                else:
                    await manager.send_personal_message({
                        "type": "error",
                        "message": f"Unknown message type: {message_type}"
                    }, websocket)
                    
            except json.JSONDecodeError:
                await manager.send_personal_message({
                    "type": "error",
                    "message": "Invalid JSON format"
                }, websocket)
            except Exception as e:
                logger.error(f"WebSocket processing error: {e}")
                await manager.send_personal_message({
                    "type": "error",
                    "message": "Internal processing error"
                }, websocket)
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)
