"""
NLU Engine - Enhanced with Bus Query Processing
"""
import asyncio
import re
from typing import Dict, Any, Optional
import logging
from config.settings import settings
from fetcher.fetch_data import bus_data_fetcher

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

logger = logging.getLogger(__name__)

class NLUEngine:
    """Natural Language Understanding engine for bus queries."""
    
    def __init__(self):
        self.engine = getattr(settings, 'nlu_engine', 'gemini')
        self.timeout = getattr(settings, 'nlu_timeout', 10)
        
        self.gemini_client = None
        self._initialize_gemini()
        
        # Bus query patterns
        self.bus_patterns = {
            'location': [
                r'where is bus (\w+)',
                r'(\w+) bus location',
                r'locate bus (\w+)',
                r'bus (\w+) position'
            ],
            'search': [
                r'buses from (.+) to (.+)',
                r'find buses (.+) to (.+)',
                r'search buses (.+) (.+)'
            ]
        }
        
        logger.info(f"NLUEngine initialized - Engine: {self.engine}")
    
    def _initialize_gemini(self):
        """Initialize Gemini client."""
        if not GENAI_AVAILABLE:
            self.engine = 'rule_based'
            logger.warning("Gemini not available - using rule-based NLU")
            return
        
        try:
            if settings.is_gemini_location_enabled():
                genai.configure(api_key=settings.gemini_api_key)
                self.gemini_client = genai.GenerativeModel(settings.gemini_model)
                logger.info("Gemini NLU client initialized")
            else:
                self.engine = 'rule_based'
                logger.warning("Gemini API key not configured - using rule-based NLU")
        except Exception as e:
            logger.error(f"Gemini NLU initialization error: {e}")
            self.engine = 'rule_based'
    
    async def process_query(self, text: str, language: str = 'en') -> Dict[str, Any]:
        """Process natural language query."""
        try:
            # First try to extract intent using patterns
            intent_result = self.extract_intent(text)
            
            if intent_result['intent'] != 'unknown':
                # Process bus-specific query
                return await self.process_bus_query(intent_result, language)
            else:
                # Use Gemini for general understanding
                if self.engine == 'gemini' and self.gemini_client:
                    return await self.gemini_process(text, language)
                else:
                    return self.rule_based_process(text, language)
                    
        except Exception as e:
            logger.error(f"NLU processing error: {e}")
            return {
                "success": False,
                "error": str(e),
                "intent": "error",
                "response": "I'm sorry, I couldn't understand your request."
            }
    
    def extract_intent(self, text: str) -> Dict[str, Any]:
        """Extract intent from text using patterns."""
        text_lower = text.lower()
        
        # Check for bus location queries
        for pattern in self.bus_patterns['location']:
            match = re.search(pattern, text_lower)
            if match:
                return {
                    "intent": "bus_location",
                    "entities": {"bus_number": match.group(1)},
                    "confidence": 0.9
                }
        
        # Check for bus search queries
        for pattern in self.bus_patterns['search']:
            match = re.search(pattern, text_lower)
            if match:
                return {
                    "intent": "bus_search",
                    "entities": {
                        "start": match.group(1).strip(),
                        "end": match.group(2).strip() if len(match.groups()) > 1 else None
                    },
                    "confidence": 0.8
                }
        
        # Check for greetings
        greetings = ['hello', 'hi', 'hey', 'namaste', 'sat sri akal']
        if any(greeting in text_lower for greeting in greetings):
            return {
                "intent": "greeting",
                "entities": {},
                "confidence": 0.9
            }
        
        return {
            "intent": "unknown",
            "entities": {},
            "confidence": 0.0
        }
    
    async def process_bus_query(self, intent_result: Dict[str, Any], language: str) -> Dict[str, Any]:
        """Process bus-specific queries."""
        try:
            intent = intent_result['intent']
            entities = intent_result['entities']
            
            if intent == 'bus_location':
                # Get bus location
                bus_number = entities.get('bus_number')
                if bus_number:
                    bus_data = await bus_data_fetcher.get_bus_by_name(bus_number)
                    return self.format_bus_location_response(bus_data, language)
                else:
                    return self.error_response("Bus number not found", language)
            
            elif intent == 'bus_search':
                # Search buses
                start = entities.get('start')
                end = entities.get('end')
                search_data = await bus_data_fetcher.search_buses(start, end)
                return self.format_bus_search_response(search_data, language)
            
            elif intent == 'greeting':
                return self.greeting_response(language)
            
            else:
                return self.error_response("Unknown query type", language)
                
        except Exception as e:
            logger.error(f"Bus query processing error: {e}")
            return self.error_response(str(e), language)
    
    def format_bus_location_response(self, bus_data: Dict[str, Any], language: str) -> Dict[str, Any]:
        """Format bus location response."""
        try:
            if bus_data.get('success') and bus_data.get('bus'):
                bus = bus_data['bus']
                location = bus['current_location']['location_name']
                bus_number = bus['bus_number']
                
                responses = {
                    'en': f"Bus {bus_number} is currently at {location}.",
                    'hi': f"बस {bus_number} अभी {location} पर है।",
                    'pa': f"ਬੱਸ {bus_number} ਹੁਣ {location} 'ਤੇ ਹੈ।"
                }
                
                return {
                    "success": True,
                    "intent": "bus_location",
                    "response": responses.get(language, responses['en']),
                    "data": bus
                }
            else:
                return self.error_response("Bus not found", language)
                
        except Exception as e:
            return self.error_response(str(e), language)
    
    def format_bus_search_response(self, search_data: Dict[str, Any], language: str) -> Dict[str, Any]:
        """Format bus search response."""
        try:
            if search_data.get('success') and search_data.get('buses'):
                buses = search_data['buses']
                count = len(buses)
                
                if count == 0:
                    responses = {
                        'en': "No buses found for your route.",
                        'hi': "आपके रूट के लिए कोई बस नहीं मिली।",
                        'pa': "ਤੁਹਾਡੇ ਰੂਟ ਲਈ ਕੋਈ ਬੱਸ ਨਹੀਂ ਮਿਲੀ।"
                    }
                elif count == 1:
                    bus = buses[0]
                    location = bus['current_location']['location_name']
                    responses = {
                        'en': f"Found 1 bus: {bus['bus_number']} currently at {location}.",
                        'hi': f"1 बस मिली: {bus['bus_number']} अभी {location} पर है।",
                        'pa': f"1 ਬੱਸ ਮਿਲੀ: {bus['bus_number']} ਹੁਣ {location} 'ਤੇ ਹੈ।"
                    }
                else:
                    responses = {
                        'en': f"Found {count} buses available for your route.",
                        'hi': f"आपके रूट के लिए {count} बसें उपलब्ध हैं।",
                        'pa': f"ਤੁਹਾਡੇ ਰੂਟ ਲਈ {count} ਬੱਸਾਂ ਉਪਲਬਧ ਹਨ।"
                    }
                
                return {
                    "success": True,
                    "intent": "bus_search",
                    "response": responses.get(language, responses['en']),
                    "data": buses
                }
            else:
                return self.error_response("No buses found", language)
                
        except Exception as e:
            return self.error_response(str(e), language)
    
    def greeting_response(self, language: str) -> Dict[str, Any]:
        """Generate greeting response."""
        responses = {
            'en': "Hello! I'm your AI bus assistant. How can I help you with bus information today?",
            'hi': "नमस्ते! मैं आपका AI बस असिस्टेंट हूँ। आज बस की जानकारी के लिए मैं आपकी कैसे मदद कर सकता हूँ?",
            'pa': "ਸਤ ਸ੍ਰੀ ਅਕਾਲ! ਮੈਂ ਤੁਹਾਡਾ AI ਬੱਸ ਅਸਿਸਟੈਂਟ ਹਾਂ। ਅੱਜ ਬੱਸ ਦੀ ਜਾਣਕਾਰੀ ਲਈ ਮੈਂ ਤੁਹਾਡੀ ਕਿਵੇਂ ਮਦਦ ਕਰ ਸਕਦਾ ਹਾਂ?"
        }
        
        return {
            "success": True,
            "intent": "greeting",
            "response": responses.get(language, responses['en'])
        }
    
    def error_response(self, error: str, language: str) -> Dict[str, Any]:
        """Generate error response."""
        responses = {
            'en': "I'm sorry, I couldn't find that information. Please try asking about a specific bus number or route.",
            'hi': "मुझे खुशी है, मुझे वह जानकारी नहीं मिली। कृपया किसी विशिष्ट बस नंबर या रूट के बारे में पूछने की कोशिश करें।",
            'pa': "ਮੈਨੂੰ ਅਫ਼ਸੋਸ ਹੈ, ਮੈਨੂੰ ਉਹ ਜਾਣਕਾਰੀ ਨਹੀਂ ਮਿਲੀ। ਕਿਰਪਾ ਕਰਕੇ ਕਿਸੇ ਖਾਸ ਬੱਸ ਨੰਬਰ ਜਾਂ ਰੂਟ ਬਾਰੇ ਪੁੱਛਣ ਦੀ ਕੋਸ਼ਿਸ਼ ਕਰੋ।"
        }
        
        return {
            "success": False,
            "intent": "error",
            "response": responses.get(language, responses['en']),
            "error": error
        }
    
    async def gemini_process(self, text: str, language: str) -> Dict[str, Any]:
        """Process query using Gemini."""
        try:
            prompt = f"""
You are an AI bus assistant. The user asked: "{text}"

Analyze this query and respond in {language} language.
If it's about bus information, provide a helpful response.
If it's a greeting, respond warmly.
Keep the response concise and natural.

Respond with only the text response, nothing else.
"""
            
            response = self.gemini_client.generate_content(prompt)
            
            if response and response.text:
                return {
                    "success": True,
                    "intent": "general",
                    "response": response.text.strip(),
                    "engine": "gemini"
                }
            else:
                return self.rule_based_process(text, language)
                
        except Exception as e:
            logger.error(f"Gemini processing error: {e}")
            return self.rule_based_process(text, language)
    
    def rule_based_process(self, text: str, language: str) -> Dict[str, Any]:
        """Fallback rule-based processing."""
        responses = {
            'en': "I understand you're asking about buses. Please ask about a specific bus number like 'Where is bus 101A?' or search for routes.",
            'hi': "मैं समझता हूँ कि आप बसों के बारे में पूछ रहे हैं। कृपया किसी विशिष्ट बस नंबर के बारे में पूछें जैसे 'बस 101A कहाँ है?' या रूट खोजें।",
            'pa': "ਮੈਂ ਸਮਝਦਾ ਹਾਂ ਕਿ ਤੁਸੀਂ ਬੱਸਾਂ ਬਾਰੇ ਪੁੱਛ ਰਹੇ ਹੋ। ਕਿਰਪਾ ਕਰਕੇ ਕਿਸੇ ਖਾਸ ਬੱਸ ਨੰਬਰ ਬਾਰੇ ਪੁੱਛੋ ਜਿਵੇਂ 'ਬੱਸ 101A ਕਿੱਥੇ ਹੈ?' ਜਾਂ ਰੂਟ ਖੋਜੋ।"
        }
        
        return {
            "success": True,
            "intent": "general",
            "response": responses.get(language, responses['en']),
            "engine": "rule_based"
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get NLU status."""
        return {
            "engine": self.engine,
            "gemini_available": GENAI_AVAILABLE and self.gemini_client is not None,
            "supported_intents": ["bus_location", "bus_search", "greeting", "general"],
            "timeout": self.timeout
        }

# Global instance
nlu_engine = NLUEngine()
