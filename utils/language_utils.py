"""
Enhanced Language Detection Utilities - Production Ready
Comprehensive multilingual support with enterprise features
"""
import re
import logging
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
from config.settings import settings

# Try to import external language detection libraries
try:
    from langdetect import detect, LangDetectError
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

logger = logging.getLogger(__name__)

class LanguageDetector:
    """Enhanced language detector with enterprise features and comprehensive multilingual support."""
    
    def __init__(self):
        # Core configuration from settings
        self.supported_languages = getattr(settings, 'supported_languages', ["en", "hi", "pa"])
        self.default_language = getattr(settings, 'default_language', "en")
        self.enable_language_detection = getattr(settings, 'enable_language_detection', True)
        
        # Enhanced language metadata
        self.language_info = {
            "en": {
                "name": "English",
                "native_name": "English",
                "script": "latin",
                "direction": "ltr",
                "tts_voice": getattr(settings, 'tts_voice_en', 'en-US-AriaNeural'),
                "locale": "en-US",
                "iso_codes": ["en", "eng"],
                "confidence_boost": 0.1  # Boost for common fallback language
            },
            "hi": {
                "name": "Hindi",
                "native_name": "हिंदी",
                "script": "devanagari",
                "direction": "ltr",
                "tts_voice": getattr(settings, 'tts_voice_hi', 'hi-IN-SwaraNeural'),
                "locale": "hi-IN",
                "iso_codes": ["hi", "hin"],
                "confidence_boost": 0.0
            },
            "pa": {
                "name": "Punjabi",
                "native_name": "ਪੰਜਾਬੀ",
                "script": "gurmukhi",
                "direction": "ltr", 
                "tts_voice": getattr(settings, 'tts_voice_pa', 'pa-IN-GulNeural'),
                "locale": "pa-IN",
                "iso_codes": ["pa", "pan"],
                "confidence_boost": 0.0
            }
        }
        
        # Enhanced detection patterns with context awareness
        self.language_patterns = {
            "hi": {
                "keywords": [
                    # Common Hindi words
                    r'\b(कहाँ|कहा|क्या|कैसे|कब|कौन|किस|है|हैं|था|थी|होगा|होगी)\b',
                    # Question words
                    r'\b(क्या|कैसे|कब|कहाँ|कौन|किसका|किसकी|क्यों)\b',
                    # Bus-related Hindi terms
                    r'\b(बस|गाड़ी|सवारी|स्टेशन|अड्डा|रुकना|चलना|जाना|आना|ठहरना)\b',
                    # Common verbs and adjectives
                    r'\b(करना|होना|देना|लेना|आना|जाना|रहना|चलना|मिलना|दिखना|अच्छा|बुरा|बड़ा|छोटा|नया|पुराना)\b',
                    # Time and location
                    r'\b(समय|वक्त|जगह|स्थान|यहाँ|वहाँ|कहीं|सब|सभी|कुछ|कोई)\b',
                    # Postpositions
                    r'\b(में|पर|से|को|तक|का|के|की|ने|द्वारा)\b'
                ],
                "script_pattern": r'[\u0900-\u097F]+',
                "numbers": r'[०-९]+',
                "punctuation": r'[।॥]'
            },
            "pa": {
                "keywords": [
                    # Common Punjabi words
                    r'\b(ਕਿੱਥੇ|ਕਿਹਾ|ਕੀ|ਕਿਵੇਂ|ਕਦੋਂ|ਕੌਣ|ਹੈ|ਹਨ|ਸੀ|ਹੋਵੇਗਾ|ਹੋਵੇਗੀ)\b',
                    # Question words
                    r'\b(ਕੀ|ਕਿਵੇਂ|ਕਦੋਂ|ਕਿੱਥੇ|ਕੌਣ|ਕਿਸਦਾ|ਕਿਸਦੀ|ਕਿਉਂ)\b',
                    # Bus-related Punjabi terms
                    r'\b(ਬੱਸ|ਗੱਡੀ|ਸਵਾਰੀ|ਸਟੇਸ਼ਨ|ਅੱਡਾ|ਰੁਕਣਾ|ਚੱਲਣਾ|ਜਾਣਾ|ਆਉਣਾ|ਠਹਿਰਣਾ)\b',
                    # Common verbs and adjectives
                    r'\b(ਕਰਨਾ|ਹੋਣਾ|ਦੇਣਾ|ਲੈਣਾ|ਆਉਣਾ|ਜਾਣਾ|ਰਹਿਣਾ|ਚੱਲਣਾ|ਮਿਲਣਾ|ਦਿਸਣਾ|ਚੰਗਾ|ਮਾੜਾ|ਵੱਡਾ|ਛੋਟਾ|ਨਵਾਂ|ਪੁਰਾਣਾ)\b',
                    # Time and location
                    r'\b(ਸਮਾਂ|ਵਕਤ|ਥਾਂ|ਸਥਾਨ|ਇੱਥੇ|ਉੱਥੇ|ਕਿੱਤੇ|ਸਭ|ਸਾਰੇ|ਕੁਝ|ਕੋਈ)\b',
                    # Postpositions
                    r'\b(ਵਿੱਚ|ਤੇ|ਤੋਂ|ਨੂੰ|ਤੱਕ|ਦਾ|ਦੇ|ਦੀ|ਨੇ|ਦੁਆਰਾ)\b'
                ],
                "script_pattern": r'[\u0A00-\u0A7F]+',
                "numbers": r'[੦-੯]+',
                "punctuation": r'[।॥]'
            },
            "en": {
                "keywords": [
                    # Common English words
                    r'\b(where|what|how|when|who|which|is|are|was|were|will|would)\b',
                    # Question words
                    r'\b(what|how|when|where|who|whose|why|which)\b',
                    # Bus-related English terms
                    r'\b(bus|vehicle|transport|station|stop|terminal|running|moving|going|coming|waiting)\b',
                    # Common verbs and adjectives  
                    r'\b(go|come|run|stop|wait|get|take|give|have|make|good|bad|big|small|new|old)\b',
                    # Time and location
                    r'\b(time|place|location|here|there|somewhere|all|some|any|this|that)\b',
                    # Prepositions
                    r'\b(in|on|at|from|to|by|with|for|of|about)\b'
                ],
                "script_pattern": r'[a-zA-Z]+',
                "numbers": r'[0-9]+',
                "punctuation": r'[.!?;:,]'
            }
        }
        
        # Compile patterns for performance
        self.compiled_patterns = {}
        for lang, patterns in self.language_patterns.items():
            self.compiled_patterns[lang] = {
                "keywords": [re.compile(pattern, re.IGNORECASE | re.UNICODE) for pattern in patterns["keywords"]],
                "script": re.compile(patterns["script_pattern"], re.UNICODE),
                "numbers": re.compile(patterns["numbers"], re.UNICODE),
                "punctuation": re.compile(patterns["punctuation"], re.UNICODE)
            }
        
        # Statistics for monitoring
        self.stats = {
            "total_detections": 0,
            "language_counts": {lang: 0 for lang in self.supported_languages},
            "method_usage": {
                "script_detection": 0,
                "pattern_detection": 0,
                "langdetect_library": 0,
                "gemini_ai": 0,
                "default_fallback": 0
            },
            "average_confidence": 0.0,
            "last_detection_time": None
        }
        
        # Initialize Gemini for advanced detection
        self.gemini_client = None
        if GENAI_AVAILABLE and hasattr(settings, 'gemini_api_key') and settings.gemini_api_key:
            try:
                genai.configure(api_key=settings.gemini_api_key)
                self.gemini_client = genai.GenerativeModel('gemini-1.5-flash')
                logger.info("✅ Gemini initialized for advanced language detection")
            except Exception as e:
                logger.warning(f"⚠️ Gemini language detection init failed: {e}")
        
        logger.info(f"✅ LanguageDetector initialized")
        logger.info(f"🌍 Supported languages: {', '.join(self.supported_languages)}")
        logger.info(f"🔍 Detection methods: Script, Pattern, {'LangDetect' if LANGDETECT_AVAILABLE else 'None'}, {'Gemini' if self.gemini_client else 'None'}")
    
    def detect_language(self, text: str, confidence_threshold: float = 0.6) -> str:
        """
        Enhanced language detection with multiple methods and confidence scoring.
        Returns the most likely language code.
        """
        try:
            # Update statistics
            self.stats["total_detections"] += 1
            self.stats["last_detection_time"] = datetime.now().isoformat()
            
            if not self.enable_language_detection:
                return self.default_language
            
            if not text or not text.strip():
                self.stats["method_usage"]["default_fallback"] += 1
                return self.default_language
            
            text = text.strip()
            
            # Get confidence scores for all methods
            detection_results = self._get_comprehensive_detection(text)
            
            # Find the language with highest confidence
            best_language = self.default_language
            best_confidence = 0.0
            
            for lang in self.supported_languages:
                total_confidence = sum(detection_results[method].get(lang, 0) for method in detection_results)
                # Apply language-specific confidence boost
                total_confidence += self.language_info[lang]["confidence_boost"]
                
                if total_confidence > best_confidence and total_confidence >= confidence_threshold:
                    best_confidence = total_confidence
                    best_language = lang
            
            # Update statistics
            self.stats["language_counts"][best_language] += 1
            
            # Update average confidence
            total_detections = self.stats["total_detections"]
            current_avg = self.stats["average_confidence"]
            self.stats["average_confidence"] = (current_avg * (total_detections - 1) + best_confidence) / total_detections
            
            logger.debug(f"🔍 Language detected: '{text[:30]}...' → {best_language} (confidence: {best_confidence:.3f})")
            
            return best_language
            
        except Exception as e:
            logger.error(f"❌ Language detection error: {e}")
            self.stats["method_usage"]["default_fallback"] += 1
            return self.default_language
    
    def _get_comprehensive_detection(self, text: str) -> Dict[str, Dict[str, float]]:
        """Get detection results from all available methods."""
        results = {}
        
        # Method 1: Script-based detection
        results["script"] = self._detect_by_script(text)
        
        # Method 2: Pattern-based detection  
        results["patterns"] = self._detect_by_patterns(text)
        
        # Method 3: LangDetect library
        if LANGDETECT_AVAILABLE:
            results["langdetect"] = self._detect_by_langdetect(text)
        
        # Method 4: Gemini AI (for complex cases)
        if self.gemini_client and len(text) > 10:
            results["gemini"] = self._detect_by_gemini(text)
        
        return results
    
    def _detect_by_script(self, text: str) -> Dict[str, float]:
        """Enhanced script-based detection with confidence scoring."""
        try:
            self.stats["method_usage"]["script_detection"] += 1
            
            script_scores = {"en": 0.0, "hi": 0.0, "pa": 0.0}
            total_chars = len(re.sub(r'\s+', '', text))
            
            if total_chars == 0:
                return script_scores
            
            for lang in self.supported_languages:
                if lang in self.compiled_patterns:
                    # Count script characters
                    script_matches = self.compiled_patterns[lang]["script"].findall(text)
                    script_char_count = sum(len(match) for match in script_matches)
                    
                    # Count numbers in script
                    number_matches = self.compiled_patterns[lang]["numbers"].findall(text)
                    number_char_count = sum(len(match) for match in number_matches)
                    
                    # Count script-specific punctuation
                    punct_matches = self.compiled_patterns[lang]["punctuation"].findall(text)
                    punct_count = len(punct_matches)
                    
                    # Calculate confidence
                    script_ratio = script_char_count / total_chars
                    number_boost = min(0.2, number_char_count / total_chars)
                    punct_boost = min(0.1, punct_count / len(text.split()))
                    
                    script_scores[lang] = script_ratio + number_boost + punct_boost
            
            return script_scores
            
        except Exception as e:
            logger.warning(f"⚠️ Script detection error: {e}")
            return {"en": 0.0, "hi": 0.0, "pa": 0.0}
    
    def _detect_by_patterns(self, text: str) -> Dict[str, float]:
        """Enhanced pattern-based detection with weighted scoring."""
        try:
            self.stats["method_usage"]["pattern_detection"] += 1
            
            pattern_scores = {"en": 0.0, "hi": 0.0, "pa": 0.0}
            text_words = len(text.split())
            
            if text_words == 0:
                return pattern_scores
            
            for lang in self.supported_languages:
                if lang in self.compiled_patterns:
                    total_matches = 0
                    
                    # Count keyword matches with different weights
                    for pattern in self.compiled_patterns[lang]["keywords"]:
                        matches = pattern.findall(text)
                        total_matches += len(matches)
                    
                    # Normalize by text length
                    pattern_scores[lang] = min(1.0, total_matches / text_words)
            
            return pattern_scores
            
        except Exception as e:
            logger.warning(f"⚠️ Pattern detection error: {e}")
            return {"en": 0.0, "hi": 0.0, "pa": 0.0}
    
    def _detect_by_langdetect(self, text: str) -> Dict[str, float]:
        """Detection using langdetect library with error handling."""
        try:
            self.stats["method_usage"]["langdetect_library"] += 1
            
            detected = detect(text)
            scores = {"en": 0.0, "hi": 0.0, "pa": 0.0}
            
            # Map langdetect results to our supported languages
            if detected in self.supported_languages:
                scores[detected] = 0.8  # High confidence for library detection
            elif detected == "gu":  # Gujarati often misidentified as Hindi
                scores["hi"] = 0.5
            elif detected in ["en", "eng"]:
                scores["en"] = 0.8
            
            return scores
            
        except (LangDetectError, Exception) as e:
            logger.debug(f"LangDetect failed: {e}")
            return {"en": 0.0, "hi": 0.0, "pa": 0.0}
    
    def _detect_by_gemini(self, text: str) -> Dict[str, float]:
        """Advanced language detection using Gemini AI."""
        try:
            self.stats["method_usage"]["gemini_ai"] += 1
            
            prompt = f"""
Detect the language of this text: "{text}"

The text is related to bus/transportation queries. Possible languages:
- English (en)
- Hindi (hi) - हिंदी  
- Punjabi (pa) - ਪੰਜਾਬੀ

Respond with only the language code (en/hi/pa) and confidence (0-1), format: "code:confidence"

Examples:
- "Where is bus 101?" → "en:0.9"
- "बस कहाँ है?" → "hi:0.9"  
- "ਬੱਸ ਕਿੱਥੇ ਹੈ?" → "pa:0.9"

Response:"""
            
            response = self.gemini_client.generate_content(prompt)
            
            if response and response.text:
                result = response.text.strip()
                
                # Parse "lang:confidence" format
                if ":" in result:
                    lang_code, confidence_str = result.split(":", 1)
                    lang_code = lang_code.strip().lower()
                    
                    try:
                        confidence = float(confidence_str.strip())
                        
                        if lang_code in self.supported_languages and 0 <= confidence <= 1:
                            scores = {"en": 0.0, "hi": 0.0, "pa": 0.0}
                            scores[lang_code] = confidence
                            return scores
                    except ValueError:
                        pass
            
            return {"en": 0.0, "hi": 0.0, "pa": 0.0}
            
        except Exception as e:
            logger.debug(f"Gemini detection failed: {e}")
            return {"en": 0.0, "hi": 0.0, "pa": 0.0}
    
    def get_language_confidence(self, text: str) -> Dict[str, float]:
        """Get detailed confidence scores for each supported language."""
        try:
            if not text or not text.strip():
                return {lang: 0.0 for lang in self.supported_languages}
            
            detection_results = self._get_comprehensive_detection(text)
            
            # Combine all method scores
            combined_scores = {}
            for lang in self.supported_languages:
                total_score = 0.0
                method_count = 0
                
                for method, scores in detection_results.items():
                    if lang in scores:
                        total_score += scores[lang]
                        method_count += 1
                
                # Average score with confidence boost
                avg_score = total_score / max(method_count, 1)
                combined_scores[lang] = min(1.0, avg_score + self.language_info[lang]["confidence_boost"])
            
            return combined_scores
            
        except Exception as e:
            logger.error(f"❌ Confidence calculation error: {e}")
            return {lang: 0.0 for lang in self.supported_languages}
    
    def validate_language(self, language_code: str) -> bool:
        """Validate if language code is supported."""
        return language_code.lower() in self.supported_languages
    
    def normalize_language_code(self, lang_input: str) -> str:
        """Normalize various language inputs to standard codes."""
        if not lang_input:
            return self.default_language
        
        lang_input = lang_input.lower().strip()
        
        # Direct mapping
        if lang_input in self.supported_languages:
            return lang_input
        
        # Name-based mapping
        name_mappings = {
            "english": "en",
            "hindi": "hi", 
            "punjabi": "pa",
            "हिंदी": "hi",
            "ਪੰਜਾਬੀ": "pa",
            "en-us": "en",
            "hi-in": "hi",
            "pa-in": "pa",
            "eng": "en",
            "hin": "hi",
            "pan": "pa"
        }
        
        return name_mappings.get(lang_input, self.default_language)
    
    def get_language_info(self, language_code: str) -> Dict[str, Any]:
        """Get comprehensive information about a language."""
        normalized_code = self.normalize_language_code(language_code)
        return self.language_info.get(normalized_code, self.language_info[self.default_language]).copy()
    
    def get_language_name(self, language_code: str, native: bool = False) -> str:
        """Get language name in English or native script."""
        info = self.get_language_info(language_code)
        return info.get("native_name" if native else "name", "Unknown")
    
    def get_tts_voice(self, language_code: str) -> str:
        """Get TTS voice identifier for a language."""
        info = self.get_language_info(language_code)
        return info.get("tts_voice", "en-US-AriaNeural")
    
    def get_supported_languages_info(self) -> List[Dict[str, Any]]:
        """Get detailed information about all supported languages."""
        return [
            {
                "code": lang,
                "name": info["name"],
                "native_name": info["native_name"],
                "script": info["script"],
                "direction": info["direction"],
                "tts_voice": info["tts_voice"],
                "locale": info["locale"],
                "iso_codes": info["iso_codes"]
            }
            for lang, info in self.language_info.items()
            if lang in self.supported_languages
        ]
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive language detector status."""
        return {
            "supported_languages": self.supported_languages,
            "default_language": self.default_language,
            "detection_enabled": self.enable_language_detection,
            "detection_methods": {
                "script_based": True,
                "pattern_based": True,
                "langdetect_library": LANGDETECT_AVAILABLE,
                "gemini_ai": self.gemini_client is not None
            },
            "statistics": self.stats,
            "performance": {
                "detection_accuracy": self.stats["average_confidence"],
                "most_detected_language": max(
                    self.stats["language_counts"], 
                    key=self.stats["language_counts"].get
                ) if any(self.stats["language_counts"].values()) else self.default_language,
                "detection_distribution": self.stats["language_counts"]
            },
            "configuration": {
                "total_patterns": sum(
                    len(patterns["keywords"]) for patterns in self.language_patterns.values()
                ),
                "confidence_threshold": 0.6
            }
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform language detector health check."""
        try:
            # Test detection with sample texts
            test_cases = {
                "en": "Where is bus number 101?",
                "hi": "बस नंबर 101 कहाँ है?",
                "pa": "ਬੱਸ ਨੰਬਰ 101 ਕਿੱਥੇ ਹੈ?"
            }
            
            results = {}
            for expected_lang, test_text in test_cases.items():
                detected = self.detect_language(test_text)
                results[expected_lang] = detected == expected_lang
            
            accuracy = sum(results.values()) / len(results) * 100
            
            return {
                "status": "healthy" if accuracy >= 66.7 else "degraded",
                "test_accuracy_percent": round(accuracy, 1),
                "test_results": results,
                "methods_available": {
                    "script_detection": True,
                    "pattern_detection": True,
                    "langdetect": LANGDETECT_AVAILABLE,
                    "gemini": self.gemini_client is not None
                },
                "last_check": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "last_check": datetime.now().isoformat()
            }
    
    def reset_statistics(self):
        """Reset language detection statistics."""
        self.stats = {
            "total_detections": 0,
            "language_counts": {lang: 0 for lang in self.supported_languages},
            "method_usage": {
                "script_detection": 0,
                "pattern_detection": 0,
                "langdetect_library": 0,
                "gemini_ai": 0,
                "default_fallback": 0
            },
            "average_confidence": 0.0,
            "last_detection_time": None
        }
        logger.info("📊 Language detection statistics reset")

# Global instance with backward compatibility
language_detector = LanguageDetector()

# Legacy aliases for backward compatibility
language_utils = language_detector
def detect_language(text: str) -> str:
    return language_detector.detect_language(text)

def get_language_info(language_code: str) -> Dict[str, Any]:
    return language_detector.get_language_info(language_code)

def validate_language(language_code: str) -> bool:
    return language_detector.validate_language(language_code)

def normalize_language_code(lang_input: str) -> str:
    return language_detector.normalize_language_code(lang_input)

def get_language_name(language_code: str, native: bool = False) -> str:
    return language_detector.get_language_name(language_code, native)

def get_tts_voice(language_code: str) -> str:
    return language_detector.get_tts_voice(language_code)

def get_supported_languages_info() -> List[Dict[str, Any]]:
    return language_detector.get_supported_languages_info()
