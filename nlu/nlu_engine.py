"""
NLU Engine - Enhanced with Natural Bus-Focused Responses
No more ChatGPT-style responses - Direct, natural bus assistant
"""
import asyncio
import re
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime
from config.settings import settings
from fetcher.fetch_data import bus_data_fetcher

# Import language utils with fallback
try:
    from utils.language_utils import language_utils
except ImportError:
    language_utils = None

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

logger = logging.getLogger(__name__)

class NLUEngine:
    """Enhanced Natural Language Understanding engine with natural bus responses."""
    
    def __init__(self):
        self.engine = getattr(settings, 'nlu_engine', 'gemini')
        self.timeout = getattr(settings, 'nlu_timeout', 10)
        
        self.gemini_client = None
        self._initialize_gemini()
        
        # Enhanced multilingual bus query patterns
        self.bus_patterns = {
            'location': {
                'en': [
                    r'where is bus (\w+)',
                    r'(\w+) bus location',
                    r'locate bus (\w+)', 
                    r'bus (\w+) position',
                    r'find bus (\w+)',
                    r'bus (\w+) where',
                    r'(\w+) bus where is',
                    r'show me bus (\w+)',
                    r'track bus (\w+)'
                ],
                'hi': [
                    r'बस (\w+) कहाँ है',
                    r'(\w+) बस कहाँ',
                    r'बस (\w+) की लोकेशन',
                    r'(\w+) बस का स्थान',
                    r'बस (\w+) ढूंढो',
                    r'(\w+) बस दिखाओ',
                    r'बस (\w+) कहां पर है'
                ],
                'pa': [
                    r'ਬੱਸ (\w+) ਕਿੱਥੇ ਹੈ',
                    r'(\w+) ਬੱਸ ਕਿੱਥੇ',
                    r'ਬੱਸ (\w+) ਦੀ ਲੋਕੇਸ਼ਨ',
                    r'(\w+) ਬੱਸ ਦਾ ਸਥਾਨ',
                    r'ਬੱਸ (\w+) ਲੱਭੋ',
                    r'(\w+) ਬੱਸ ਵਿਖਾਓ'
                ]
            },
            'search': {
                'en': [
                    r'buses from (.+?) to (.+?)(?:\?|$)',
                    r'find buses (.+?) to (.+?)(?:\?|$)',
                    r'search buses (.+?) to (.+?)(?:\?|$)',
                    r'(.+?) to (.+?) buses',
                    r'buses between (.+?) and (.+?)(?:\?|$)',
                    r'show buses (.+?) to (.+?)(?:\?|$)'
                ],
                'hi': [
                    r'(.+?) से (.+?) तक बस',
                    r'(.+?) से (.+?) बस',
                    r'बस (.+?) से (.+?) तक',
                    r'(.+?) से (.+?) के लिए बस',
                    r'बस खोजो (.+?) से (.+?) तक',
                    r'(.+?) और (.+?) के बीच बस'
                ],
                'pa': [
                    r'(.+?) ਤੋਂ (.+?) ਤੱਕ ਬੱਸ',
                    r'(.+?) ਤੋਂ (.+?) ਬੱਸ', 
                    r'ਬੱਸ (.+?) ਤੋਂ (.+?) ਤੱਕ',
                    r'(.+?) ਤੋਂ (.+?) ਲਈ ਬੱਸ',
                    r'ਬੱਸ ਲੱਭੋ (.+?) ਤੋਂ (.+?) ਤੱਕ'
                ]
            },
            'status': {
                'en': [
                    r'bus (\w+) status',
                    r'(\w+) bus running',
                    r'is bus (\w+) running',
                    r'bus (\w+) active'
                ],
                'hi': [
                    r'बस (\w+) स्थिति',
                    r'(\w+) बस चल रही',
                    r'बस (\w+) सक्रिय',
                    r'क्या बस (\w+) चल रही'
                ],
                'pa': [
                    r'ਬੱਸ (\w+) ਸਥਿਤੀ',
                    r'(\w+) ਬੱਸ ਚੱਲ ਰਹੀ',
                    r'ਬੱਸ (\w+) ਸਰਗਰਮ'
                ]
            },
            'greeting': {
                'en': ['hello', 'hi', 'hey', 'good morning', 'good evening', 'help'],
                'hi': ['नमस्ते', 'हैलो', 'हाय', 'सुप्रभात', 'शुभ संध्या', 'मदद'],
                'pa': ['ਸਤ ਸ੍ਰੀ ਅਕਾਲ', 'ਹੈਲੋ', 'ਹਾਏ', 'ਸੁਪ੍ਰਭਾਤ', 'ਸ਼ੁਭ ਸੰਧਿਆ', 'ਮਦਦ']
            }
        }
        
        # Statistics for monitoring
        self.stats = {
            'total_queries': 0,
            'successful_queries': 0,
            'failed_queries': 0,
            'intent_counts': {},
            'language_counts': {},
            'last_query_time': None
        }
        
        logger.info(f"✅ NLUEngine initialized - Engine: {self.engine}")
    
    def _initialize_gemini(self):
        """Initialize Gemini client with enhanced error handling."""
        if not GENAI_AVAILABLE:
            self.engine = 'rule_based'
            logger.warning("⚠️ Gemini not available - using rule-based NLU")
            return
        
        try:
            if settings.is_gemini_location_enabled():
                genai.configure(api_key=settings.gemini_api_key)
                self.gemini_client = genai.GenerativeModel(settings.gemini_model)
                logger.info("✅ Gemini NLU client initialized")
            else:
                self.engine = 'rule_based'
                logger.warning("⚠️ Gemini API key not configured - using rule-based NLU")
        except Exception as e:
            logger.error(f"❌ Gemini NLU initialization error: {e}")
            self.engine = 'rule_based'
    
    async def process_query(self, text: str, language: str = 'en') -> Dict[str, Any]:
        """Process natural language query with enhanced error handling."""
        try:
            # Update statistics
            self.stats['total_queries'] += 1
            self.stats['last_query_time'] = datetime.now().isoformat()
            self.stats['language_counts'][language] = self.stats['language_counts'].get(language, 0) + 1
            
            # Validate input
            if not text or not text.strip():
                return self.error_response("Empty query", language)
            
            text = text.strip()
            
            # Auto-detect language if language_utils available
            detected_language = language
            if language_utils:
                detected_language = language_utils.detect_language(text)
                if language_utils.validate_language(detected_language):
                    language = detected_language
                    logger.info(f"🌍 NLU detected language: {text[:30]}... → {language}")
            
            # Extract intent using enhanced patterns
            intent_result = self.extract_intent(text, language)
            
            # Update intent statistics
            intent = intent_result['intent']
            self.stats['intent_counts'][intent] = self.stats['intent_counts'].get(intent, 0) + 1
            
            if intent != 'unknown':
                # Process bus-specific query
                result = await self.process_bus_query(intent_result, language)
            else:
                # Use Gemini for general understanding
                if self.engine == 'gemini' and self.gemini_client:
                    result = await self.gemini_process(text, language)
                else:
                    result = self.rule_based_process(text, language)
            
            # Update success statistics
            if result.get('success', False):
                self.stats['successful_queries'] += 1
            else:
                self.stats['failed_queries'] += 1
            
            return result
                    
        except Exception as e:
            logger.error(f"❌ NLU processing error: {e}")
            self.stats['failed_queries'] += 1
            return {
                "success": False,
                "error": str(e),
                "intent": "error",
                "response": self.error_response("Processing error", language)['response'],
                "language": language,
                "timestamp": datetime.now().isoformat()
            }
    
    def extract_intent(self, text: str, language: str) -> Dict[str, Any]:
        """Enhanced intent extraction with multilingual support."""
        text_lower = text.lower().strip()
        
        # Check for bus location queries in detected language
        if language in self.bus_patterns['location']:
            for pattern in self.bus_patterns['location'][language]:
                match = re.search(pattern, text_lower, re.IGNORECASE | re.UNICODE)
                if match:
                    bus_number = match.group(1).upper()
                    logger.info(f"🎯 Location intent detected: {bus_number} ({language})")
                    return {
                        "intent": "bus_location",
                        "entities": {"bus_number": bus_number},
                        "confidence": 0.9,
                        "language": language
                    }
        
        # Check for bus search queries
        if language in self.bus_patterns['search']:
            for pattern in self.bus_patterns['search'][language]:
                match = re.search(pattern, text_lower, re.IGNORECASE | re.UNICODE)
                if match:
                    start = match.group(1).strip().title()
                    end = match.group(2).strip().title() if len(match.groups()) > 1 else None
                    logger.info(f"🎯 Search intent detected: {start} → {end} ({language})")
                    return {
                        "intent": "bus_search",
                        "entities": {"start": start, "end": end},
                        "confidence": 0.8,
                        "language": language
                    }
        
        # Check for bus status queries
        if language in self.bus_patterns['status']:
            for pattern in self.bus_patterns['status'][language]:
                match = re.search(pattern, text_lower, re.IGNORECASE | re.UNICODE)
                if match:
                    bus_number = match.group(1).upper()
                    logger.info(f"🎯 Status intent detected: {bus_number} ({language})")
                    return {
                        "intent": "bus_status",
                        "entities": {"bus_number": bus_number},
                        "confidence": 0.8,
                        "language": language
                    }
        
        # Check for greetings
        if language in self.bus_patterns['greeting']:
            greetings = self.bus_patterns['greeting'][language]
            if any(greeting in text_lower for greeting in greetings):
                logger.info(f"🎯 Greeting intent detected ({language})")
                return {
                    "intent": "greeting",
                    "entities": {},
                    "confidence": 0.9,
                    "language": language
                }
        
        # Fallback: check all languages if current language didn't match
        for lang in self.bus_patterns['location']:
            if lang != language:
                for pattern in self.bus_patterns['location'][lang]:
                    match = re.search(pattern, text_lower, re.IGNORECASE | re.UNICODE)
                    if match:
                        bus_number = match.group(1).upper()
                        logger.info(f"🎯 Cross-language location intent: {bus_number} ({lang})")
                        return {
                            "intent": "bus_location",
                            "entities": {"bus_number": bus_number},
                            "confidence": 0.7,
                            "language": lang
                        }
        
        logger.info(f"🤷 Unknown intent: {text[:50]}...")
        return {
            "intent": "unknown",
            "entities": {},
            "confidence": 0.0,
            "language": language
        }
    
    async def process_bus_query(self, intent_result: Dict[str, Any], language: str) -> Dict[str, Any]:
        """Process bus-specific queries with enhanced error handling."""
        try:
            intent = intent_result['intent']
            entities = intent_result['entities']
            
            if intent == 'bus_location':
                bus_number = entities.get('bus_number')
                if bus_number:
                    bus_data = await bus_data_fetcher.get_bus_by_name(bus_number)
                    return self.format_bus_location_response(bus_data, bus_number, language)
                else:
                    return self.error_response("Bus number not found", language)
            
            elif intent == 'bus_search':
                start = entities.get('start')
                end = entities.get('end')
                if start:
                    search_data = await bus_data_fetcher.search_buses(start, end)
                    return self.format_bus_search_response(search_data, start, end, language)
                else:
                    return self.error_response("Start location not specified", language)
            
            elif intent == 'bus_status':
                bus_number = entities.get('bus_number')
                if bus_number:
                    bus_data = await bus_data_fetcher.get_bus_by_name(bus_number)
                    return self.format_bus_status_response(bus_data, bus_number, language)
                else:
                    return self.error_response("Bus number not found", language)
            
            elif intent == 'greeting':
                return self.greeting_response(language)
            
            else:
                return self.error_response("Unknown query type", language)
                
        except Exception as e:
            logger.error(f"❌ Bus query processing error: {e}")
            return self.error_response(str(e), language)
    
    def format_bus_location_response(self, bus_data: Dict[str, Any], bus_number: str, language: str) -> Dict[str, Any]:
        """Format natural bus location response - NO ChatGPT style."""
        try:
            if bus_data.get('success') and bus_data.get('bus'):
                bus = bus_data['bus']
                location = bus['current_location']['location_name']
                
                # NATURAL, DIRECT responses - NO AI-speak
                responses = {
                    'en': f"Bus {bus_number} is at {location}.",
                    'hi': f"बस {bus_number} {location} पर है।",
                    'pa': f"ਬੱਸ {bus_number} {location} 'ਤੇ ਹੈ।"
                }
                
                return {
                    "success": True,
                    "intent": "bus_location",
                    "response": responses.get(language, responses['en']),
                    "data": {"bus": bus, "query_bus_number": bus_number},
                    "language": language,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                # Simple, direct error messages - NO AI-speak
                responses = {
                    'en': f"Bus {bus_number} not found. Check the bus number.",
                    'hi': f"बस {bus_number} नहीं मिली। बस नंबर चेक करें।",
                    'pa': f"ਬੱਸ {bus_number} ਨਹੀਂ ਮਿਲੀ। ਬੱਸ ਨੰਬਰ ਚੈੱਕ ਕਰੋ।"
                }
                
                return {
                    "success": False,
                    "intent": "bus_location", 
                    "response": responses.get(language, responses['en']),
                    "error": "Bus not found",
                    "data": {"query_bus_number": bus_number},
                    "language": language,
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"❌ Bus location formatting error: {e}")
            # Simple error response - NO AI-speak
            responses = {
                'en': f"Can't check bus {bus_number} right now.",
                'hi': f"अभी बस {bus_number} की जानकारी नहीं मिल रही।",
                'pa': f"ਅਭੀ ਬੱਸ {bus_number} ਦੀ ਜਾਣਕਾਰੀ ਨਹੀਂ ਮਿਲ ਰਹੀ।"
            }
            return {
                "success": False,
                "intent": "bus_location",
                "response": responses.get(language, responses['en']),
                "error": str(e),
                "language": language,
                "timestamp": datetime.now().isoformat()
            }
    
    def format_bus_search_response(self, search_data: Dict[str, Any], start: str, end: str, language: str) -> Dict[str, Any]:
        """Format natural bus search response - NO ChatGPT style."""
        try:
            if search_data.get('success') and search_data.get('buses'):
                buses = search_data['buses']
                count = len(buses)
                
                if count == 0:
                    # Direct, simple responses
                    responses = {
                        'en': f"No buses from {start} to {end}.",
                        'hi': f"{start} से {end} तक कोई बस नहीं।",
                        'pa': f"{start} ਤੋਂ {end} ਤੱਕ ਕੋਈ ਬੱਸ ਨਹੀਂ।"
                    }
                    
                    return {
                        "success": False,
                        "intent": "bus_search",
                        "response": responses.get(language, responses['en']),
                        "data": {"buses": [], "query": {"start": start, "end": end}},
                        "language": language,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                elif count == 1:
                    bus = buses[0]
                    location = bus['current_location']['location_name']
                    responses = {
                        'en': f"Bus {bus['bus_number']} found - currently at {location}.",
                        'hi': f"बस {bus['bus_number']} मिली - अभी {location} पर है।",
                        'pa': f"ਬੱਸ {bus['bus_number']} ਮਿਲੀ - ਹੁਣ {location} 'ਤੇ ਹੈ।"
                    }
                else:
                    responses = {
                        'en': f"{count} buses found from {start} to {end}.",
                        'hi': f"{start} से {end} तक {count} बसें मिलीं।",
                        'pa': f"{start} ਤੋਂ {end} ਤੱਕ {count} ਬੱਸਾਂ ਮਿਲੀਆਂ।"
                    }
                
                return {
                    "success": True,
                    "intent": "bus_search",
                    "response": responses.get(language, responses['en']),
                    "data": {"buses": buses, "total_count": count, "query": {"start": start, "end": end}},
                    "language": language,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                responses = {
                    'en': f"Bus search failed for {start} to {end}.",
                    'hi': f"{start} से {end} तक बस खोज असफल।",
                    'pa': f"{start} ਤੋਂ {end} ਤੱਕ ਬੱਸ ਖੋਜ ਅਸਫਲ।"
                }
                
                return {
                    "success": False,
                    "intent": "bus_search",
                    "response": responses.get(language, responses['en']),
                    "error": "Search failed",
                    "data": {"query": {"start": start, "end": end}},
                    "language": language,
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"❌ Bus search formatting error: {e}")
            responses = {
                'en': "Bus search not working right now.",
                'hi': "बस खोज अभी काम नहीं कर रही।",
                'pa': "ਬੱਸ ਖੋਜ ਅਭੀ ਕੰਮ ਨਹੀਂ ਕਰ ਰਹੀ।"
            }
            return {
                "success": False,
                "intent": "bus_search", 
                "response": responses.get(language, responses['en']),
                "error": str(e),
                "language": language,
                "timestamp": datetime.now().isoformat()
            }
    
    def format_bus_status_response(self, bus_data: Dict[str, Any], bus_number: str, language: str) -> Dict[str, Any]:
        """Format natural bus status response - NO ChatGPT style."""
        try:
            if bus_data.get('success') and bus_data.get('bus'):
                bus = bus_data['bus']
                status = bus.get('status', 'unknown')
                location = bus['current_location']['location_name']
                
                # Natural status responses
                if status.lower() == 'active':
                    responses = {
                        'en': f"Bus {bus_number} is running. Currently at {location}.",
                        'hi': f"बस {bus_number} चल रही है। अभी {location} पर है।",
                        'pa': f"ਬੱਸ {bus_number} ਚੱਲ ਰਹੀ ਹੈ। ਹੁਣ {location} 'ਤੇ ਹੈ।"
                    }
                else:
                    responses = {
                        'en': f"Bus {bus_number} status: {status}. Location: {location}.",
                        'hi': f"बस {bus_number} स्थिति: {status}। स्थान: {location}।",
                        'pa': f"ਬੱਸ {bus_number} ਸਥਿਤੀ: {status}। ਸਥਾਨ: {location}।"
                    }
                
                return {
                    "success": True,
                    "intent": "bus_status",
                    "response": responses.get(language, responses['en']),
                    "data": {"bus": bus, "query_bus_number": bus_number},
                    "language": language,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return self.format_bus_location_response(bus_data, bus_number, language)
                
        except Exception as e:
            return self.error_response(str(e), language)
    
    def greeting_response(self, language: str) -> Dict[str, Any]:
        """Generate natural greeting response - NO ChatGPT style."""
        # SHORT, natural greetings
        responses = {
            'en': "Hi! Ask me about buses - like 'Where is bus 101A?' or 'Buses from City Center to Airport'.",
            'hi': "नमस्ते! बसों के बारे में पूछें - जैसे 'बस 101A कहाँ है?' या 'सिटी सेंटर से एयरपोर्ट तक बसें'।",
            'pa': "ਸਤ ਸ੍ਰੀ ਅਕਾਲ! ਬੱਸਾਂ ਬਾਰੇ ਪੁੱਛੋ - ਜਿਵੇਂ 'ਬੱਸ 101A ਕਿੱਥੇ ਹੈ?' ਜਾਂ 'ਸਿਟੀ ਸੈਂਟਰ ਤੋਂ ਏਅਰਪੋਰਟ ਤੱਕ ਬੱਸਾਂ'।"
        }
        
        return {
            "success": True,
            "intent": "greeting", 
            "response": responses.get(language, responses['en']),
            "language": language,
            "timestamp": datetime.now().isoformat()
        }
    
    def error_response(self, error: str, language: str) -> Dict[str, Any]:
        """Generate natural error response - NO ChatGPT style."""
        # SHORT, helpful error messages
        responses = {
            'en': "Try asking 'Where is bus 101A?' or 'Buses from City Center to Airport'.",
            'hi': "'बस 101A कहाँ है?' या 'सिटी सेंटर से एयरपोर्ट तक बसें' पूछकर देखें।",
            'pa': "'ਬੱਸ 101A ਕਿੱਥੇ ਹੈ?' ਜਾਂ 'ਸਿਟੀ ਸੈਂਟਰ ਤੋਂ ਏਅਰਪੋਰਟ ਤੱਕ ਬੱਸਾਂ' ਪੁੱਛੋ।"
        }
        
        return {
            "success": False,
            "intent": "error",
            "response": responses.get(language, responses['en']),
            "error": error,
            "language": language,
            "timestamp": datetime.now().isoformat()
        }
    
    async def gemini_process(self, text: str, language: str) -> Dict[str, Any]:
        """Enhanced Gemini processing with bus-focused prompts."""
        try:
            prompt = f"""You are a bus information assistant. User asked: "{text}"

Instructions:
- Give SHORT, direct answers about buses only
- Don't say "I'm an AI" or "I can help you"  
- Just give the bus information directly
- If it's not about buses, say "I only help with buses"
- Respond in {language} language
- Keep it under 20 words

Examples:
User: "Where is bus 101?" → "Bus 101 is at Central Station."
User: "Weather today?" → "I only help with buses."

Response:"""
            
            response = self.gemini_client.generate_content(prompt)
            
            if response and response.text:
                clean_response = response.text.strip()
                # Remove AI-speak if present
                ai_phrases = [
                    "I'm an AI", "I can help", "I'm here to", "As an AI",
                    "I'm designed to", "My purpose is", "I'm programmed"
                ]
                for phrase in ai_phrases:
                    if phrase.lower() in clean_response.lower():
                        # Fallback to simple response
                        return self.rule_based_process(text, language)
                
                return {
                    "success": True,
                    "intent": "general",
                    "response": clean_response,
                    "engine": "gemini",
                    "language": language,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return self.rule_based_process(text, language)
                
        except Exception as e:
            logger.error(f"❌ Gemini processing error: {e}")
            return self.rule_based_process(text, language)
    
    def rule_based_process(self, text: str, language: str) -> Dict[str, Any]:
        """Enhanced fallback rule-based processing - NO ChatGPT style."""
        # SHORT, direct fallback responses
        responses = {
            'en': "Ask me about bus locations or routes.",
            'hi': "मुझसे बस की लोकेशन या रूट के बारे में पूछें।",
            'pa': "ਮੈਨੂੰ ਬੱਸ ਦੀ ਸਥਿਤੀ ਜਾਂ ਰੂਟ ਬਾਰੇ ਪੁੱਛੋ।"
        }
        
        return {
            "success": True,
            "intent": "general",
            "response": responses.get(language, responses['en']),
            "engine": "rule_based",
            "language": language,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get enhanced NLU status with statistics."""
        return {
            "engine": self.engine,
            "gemini_available": GENAI_AVAILABLE and self.gemini_client is not None,
            "language_utils_available": language_utils is not None,
            "supported_intents": ["bus_location", "bus_search", "bus_status", "greeting", "general"],
            "supported_languages": list(self.bus_patterns['location'].keys()),
            "timeout": self.timeout,
            "statistics": self.stats,
            "performance": {
                "success_rate": (
                    self.stats['successful_queries'] / max(self.stats['total_queries'], 1) * 100
                    if self.stats['total_queries'] > 0 else 0
                )
            },
            "response_style": "natural_bus_focused"
        }

# Global instance
nlu_engine = NLUEngine()
