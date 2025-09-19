"""
Enhanced Language Detection Utilities
"""
import re
from typing import Optional, Dict, Any

# Try to import langdetect
try:
    from langdetect import detect, LangDetectError
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

class LanguageDetector:
    """Enhanced language detector for multilingual support."""
    
    def __init__(self):
        self.supported_languages = ["en", "hi", "pa"]
        self.language_names = {
            "en": "English",
            "hi": "Hindi", 
            "pa": "Punjabi"
        }
        self.default_language = "en"
        
        # Enhanced keyword patterns for better detection
        self.language_patterns = {
            "hi": [
                # Hindi keywords
                r'\b(कहाँ|कहा|क्या|कैसे|कब|कौन|किस|है|हैं|का|की|के|में|पर|से|को|गया|गई|चल|रुक|आना|जाना|बस|स्टेशन|समय|स्थान|स्थिति)\b',
                # Hindi numbers
                r'[०-९]+',
                # Devanagari script
                r'[\u0900-\u097F]+'
            ],
            "pa": [
                # Punjabi keywords  
                r'\b(ਕਿੱਥੇ|ਕਿਹਾ|ਕੀ|ਕਿਵੇਂ|ਕਦੋਂ|ਕੌਣ|ਹੈ|ਹਨ|ਦਾ|ਦੀ|ਦੇ|ਵਿੱਚ|ਤੇ|ਤੋਂ|ਨੂੰ|ਗਿਆ|ਗਈ|ਚੱਲ|ਰੁਕ|ਆਉਣਾ|ਜਾਣਾ|ਬੱਸ|ਸਟੇਸ਼ਨ|ਸਮਾਂ|ਸਥਾਨ|ਸਥਿਤੀ)\b',
                # Gurmukhi script
                r'[\u0A00-\u0A7F]+'
            ],
            "en": [
                # English keywords
                r'\b(where|what|how|when|who|which|is|are|bus|station|time|location|status|running|stopped)\b'
            ]
        }
        
        # Compile patterns for efficiency
        self.compiled_patterns = {}
        for lang, patterns in self.language_patterns.items():
            self.compiled_patterns[lang] = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    
    def detect_language(self, text: str) -> str:
        """
        Detect language with enhanced pattern matching.
        Returns language code (en, hi, pa).
        """
        if not text or not text.strip():
            return self.default_language
        
        text = text.strip()
        
        # Method 1: Script-based detection (most reliable for Hindi/Punjabi)
        script_lang = self._detect_by_script(text)
        if script_lang != "en":
            return script_lang
        
        # Method 2: Pattern-based detection
        pattern_lang = self._detect_by_patterns(text)
        if pattern_lang != "en":
            return pattern_lang
        
        # Method 3: langdetect library (fallback)
        if LANGDETECT_AVAILABLE:
            try:
                detected = detect(text)
                if detected in self.supported_languages:
                    return detected
                # Map some common langdetect results
                if detected == "gu":  # Sometimes Hindi is detected as Gujarati
                    return "hi"
            except (LangDetectError, Exception):
                pass
        
        # Default to English
        return self.default_language
    
    def _detect_by_script(self, text: str) -> str:
        """Detect language based on Unicode script ranges."""
        
        # Count characters in each script
        devanagari_count = len(re.findall(r'[\u0900-\u097F]', text))
        gurmukhi_count = len(re.findall(r'[\u0A00-\u0A7F]', text))
        latin_count = len(re.findall(r'[a-zA-Z]', text))
        
        total_chars = len(re.sub(r'\s+', '', text))
        
        if total_chars == 0:
            return self.default_language
        
        # If more than 30% non-Latin characters, classify accordingly
        if devanagari_count / total_chars > 0.3:
            return "hi"
        elif gurmukhi_count / total_chars > 0.3:
            return "pa"
        elif latin_count / total_chars > 0.7:
            return "en"
        
        return self.default_language
    
    def _detect_by_patterns(self, text: str) -> str:
        """Detect language using keyword patterns."""
        
        language_scores = {"en": 0, "hi": 0, "pa": 0}
        
        for lang, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                matches = pattern.findall(text)
                language_scores[lang] += len(matches)
        
        # Find language with highest score
        max_score = max(language_scores.values())
        if max_score > 0:
            for lang, score in language_scores.items():
                if score == max_score:
                    return lang
        
        return self.default_language
    
    def get_language_confidence(self, text: str) -> Dict[str, float]:
        """Get confidence scores for each language."""
        
        if not text:
            return {"en": 1.0, "hi": 0.0, "pa": 0.0}
        
        # Script-based confidence
        devanagari_ratio = len(re.findall(r'[\u0900-\u097F]', text)) / len(text)
        gurmukhi_ratio = len(re.findall(r'[\u0A00-\u0A7F]', text)) / len(text)
        latin_ratio = len(re.findall(r'[a-zA-Z]', text)) / len(text)
        
        # Pattern-based confidence
        language_scores = {"en": 0, "hi": 0, "pa": 0}
        for lang, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                matches = pattern.findall(text)
                language_scores[lang] += len(matches)
        
        total_matches = sum(language_scores.values()) or 1
        
        return {
            "en": min(1.0, latin_ratio + (language_scores["en"] / total_matches)),
            "hi": min(1.0, devanagari_ratio + (language_scores["hi"] / total_matches)),
            "pa": min(1.0, gurmukhi_ratio + (language_scores["pa"] / total_matches))
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get language detector status."""
        return {
            "supported_languages": self.supported_languages,
            "language_names": self.language_names,
            "default_language": self.default_language,
            "langdetect_available": LANGDETECT_AVAILABLE,
            "detection_methods": ["script", "patterns", "langdetect"]
        }

# Global instance
language_detector = LanguageDetector()
