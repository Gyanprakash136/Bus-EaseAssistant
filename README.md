🚌 Bus-EaseAssistant README.md
Here's a professional README.md file similar to the BusEase Backend format, tailored for your AI Bus Assistant:

text
# Bus-EaseAssistant
**AI Voice-Powered Bus Tracking System with Natural Language Conversation**

Smart Bus Assistant built with FastAPI, Gemini AI, and advanced voice processing capabilities.

## Features

🎤 **Voice-to-Voice Conversation** - Complete speech recognition and synthesis pipeline  
🧠 **AI-Powered Intelligence** - Gemini AI for natural language understanding  
🌍 **Multilingual Support** - English, Hindi (हिंदी), Punjabi (ਪੰਜਾਬੀ)  
🚌 **Real-time Bus Tracking** - Live GPS location updates and route planning  
🗣️ **Natural Speech Processing** - Vosk STT + Edge TTS for offline capabilities  
📱 **Production-Ready API** - Comprehensive FastAPI backend with WebSocket support  
🔄 **Smart Fallbacks** - Intelligent offline responses when external services unavailable  

## API Endpoints

### Voice Processing
POST /api/v1/voice/query - Process voice/text queries with AI response
POST /api/v1/voice/process - Upload and process audio files
GET /api/v1/voice/languages - Get supported languages (en, hi, pa)
POST /api/v1/conversation/start - Initialize conversation session

text

### Bus Data Integration
GET /api/v1/bus/search?start=...&end=... - Search buses by route
GET /api/v1/bus/by-name/{name} - Get bus details by name/number
GET /api/v1/bus/{bus_id} - Get specific bus information

text

### System & Real-time
GET /health - Comprehensive system health check
WebSocket /api/v1/ws - Real-time updates and live conversation
GET /api/v1/status - Detailed component status (7/7 ready)
GET /docs - Interactive API documentation

text

## Tech Stack

**Backend:** FastAPI, Python 3.8+, Uvicorn  
**AI/ML:** Google Gemini AI, Vosk Speech Recognition, Edge TTS  
**Voice Processing:** Offline STT, Multilingual TTS, Audio Processing  
**Integration:** Bus-EaseBackend API, RESTful services  
**Real-time:** WebSocket connections, Live data streaming  
**Deployment:** Render, Railway, Heroku compatible  

## System Architecture

Frontend (Voice UI) ↔ AI Backend ↔ External APIs
↓ ↓ ↓

Voice Input - Speech-to-Text - Bus Backend

Chat Interface - NLU Processing - Gemini AI

Voice Output - Text-to-Speech - Location APIs

text

## Quick Start

### Prerequisites
- Python 3.8+
- 8GB RAM (for AI models)
- Microphone access
- Internet connection (for AI APIs)

### Installation

Clone repository
git clone https://github.com/Gyanprakash136/Bus-EaseAssistant.git
cd Bus-EaseAssistant

Create virtual environment
python -m venv venv
venv\Scripts\activate # Windows

source venv/bin/activate # macOS/Linux
Install dependencies
pip install -r requirements.txt

Setup environment variables
echo "GEMINI_API_KEY=your_gemini_api_key_here" > .env
echo "BUS_BACKEND_URL=https://bus-easebackend.onrender.com" >> .env

Run application
python main.py

text

### Usage

API Server runs on: http://localhost:10000
API Documentation: http://localhost:10000/docs
Health Check: http://localhost:10000/health
Frontend: Open frontend/ai-bus-assistant-ultimate.html in Chrome/Edge
text

## Voice Commands Examples

🇺🇸 English:
"Where is bus 101A?"
"Show me buses from City Center to Airport"
"Is the 8:30 bus running on time?"

🇮🇳 Hindi:
"बस 202B कहाँ है?"
"सिटी सेंटर से एयरपोर्ट तक बसें दिखाएं"

🇮🇳 Punjabi:
"ਬੱਸ 303C ਕਿੱਥੇ ਹੈ?"
"ਮੈਨੂੰ ਸਭ ਤੋਂ ਤੇਜ਼ ਰੂਟ ਦਿਖਾਓ"

text

## API Usage Example

// Voice Query
const response = await fetch('http://localhost:10000/api/v1/voice/query', {
method: 'POST',
headers: { 'Content-Type': 'application/json' },
body: JSON.stringify({
text: 'Where is bus 101A?',
language: 'en',
detect_language: true
})
});

const data = await response.json();
// Response: { response_text: "Bus 101A is at...", audio_url: "/static/audio/...", detected_language: "en" }

text

## Component Status

**All 7/7 Components Ready:**
- ✅ **Language Utilities** - Multilingual text processing
- ✅ **Audio Processor** - Voice input/output handling  
- ✅ **Bus Data Fetcher** - Real-time API integration
- ✅ **STT Engine** - Vosk speech recognition (offline)
- ✅ **TTS Engine** - Edge text-to-speech (multilingual)
- ✅ **NLU Engine** - Gemini natural language understanding
- ✅ **Location AI** - GPS coordinate conversion

## Deployment

### Local Development
python main.py

Runs on http://localhost:10000 with auto-reload
text

### Production Deployment
Render.com (Recommended)
Set environment variables in dashboard
Deploy directly from GitHub
Railway
railway login && railway link && railway up

Docker
docker build -t bus-ease-assistant .
docker run -p 10000:10000 bus-ease-assistant

text

## Project Structure

Bus-EaseAssistant/
├── main.py # FastAPI application entry point
├── config/ # Configuration and settings
├── api/ # API routes and WebSocket handlers
├── utils/ # Utility functions and helpers
├── stt/ # Speech-to-Text engine (Vosk)
├── tts/ # Text-to-Speech engine (Edge TTS)
├── nlu/ # Natural Language Understanding (Gemini)
├── fetcher/ # Bus data fetching and caching
├── frontend/ # Web interface files
├── static/ # Static assets and audio files
└── models/ # AI models (excluded from git)

text

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- [Vosk](https://alphacephei.com/vosk/) - Open source speech recognition
- [Google Gemini AI](https://ai.google.dev/) - Advanced language understanding
- [FastAPI](https://fastapi.tiangolo.com/) - Modern Python web framework
- [Edge TTS](https://github.com/rany2/edge-tts) - High-quality text-to-speech
- [Bus-EaseBackend](https://github.com/Sunnik-Chatterjee/Bus-EaseBackend) - External bus data integration

---

**Made with ❤️ for smarter public transportation**

*⭐ Star this repo if it helped you build something awesome!*
