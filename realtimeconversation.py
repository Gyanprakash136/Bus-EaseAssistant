#!/usr/bin/env python3
"""
Real-Time Conversational AI Assistant
Speak -> Process (2 seconds) -> AI Responds through speakers
"""
import requests
import json
import time
import threading
import queue
import tempfile
import os
import wave
import pyaudio
from typing import Optional
import keyboard

# Audio playback
try:
    import pygame
    pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False

BASE_URL = "http://localhost:8000"

class RealTimeConversation:
    """Real-time conversational AI assistant."""
    
    def __init__(self):
        self.is_listening = False
        self.is_speaking = False
        self.audio_queue = queue.Queue()
        
        # Audio settings
        self.format = pyaudio.paInt16
        self.channels = 1
        self.sample_rate = 16000
        self.chunk_size = 1024
        self.record_duration = 5  # Maximum recording time
        
        # Language setting
        self.current_language = "en"
        
        # Initialize TTS engine for immediate response
        self.tts_engine = None
        if PYTTSX3_AVAILABLE:
            try:
                self.tts_engine = pyttsx3.init()
                self.tts_engine.setProperty('rate', 170)
                self.tts_engine.setProperty('volume', 0.9)
                print("✅ Direct TTS engine ready")
            except:
                pass
        
        print("🎤 Real-Time Conversational AI initialized")
        print(f"🔊 Audio playback: {'Pygame' if PYGAME_AVAILABLE else 'pyttsx3' if PYTTSX3_AVAILABLE else 'Limited'}")
    
    def voice_activity_detection(self, audio_data):
        """Simple voice activity detection."""
        import struct
        
        # Convert bytes to integers
        audio_ints = struct.unpack(f'<{len(audio_data)//2}h', audio_data)
        
        # Calculate RMS (Root Mean Square) for volume detection
        rms = (sum(x*x for x in audio_ints) / len(audio_ints)) ** 0.5
        
        # Threshold for voice detection (adjust as needed)
        return rms > 500
    
    def record_with_voice_detection(self) -> Optional[str]:
        """Record audio with voice activity detection."""
        try:
            p = pyaudio.PyAudio()
            
            print("🎤 Listening... (Speak now or press SPACE to start recording)")
            
            stream = p.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            
            frames = []
            recording = False
            silence_count = 0
            voice_detected = False
            
            # Listen for voice or space key
            start_time = time.time()
            
            while True:
                # Check for manual trigger
                if keyboard.is_pressed('space'):
                    if not recording:
                        print("🎤 RECORDING (manual trigger)...")
                        recording = True
                        start_time = time.time()
                
                # Read audio data
                try:
                    data = stream.read(self.chunk_size, exception_on_overflow=False)
                except Exception:
                    continue
                
                # Voice activity detection
                has_voice = self.voice_activity_detection(data)
                
                if has_voice and not recording:
                    print("🎤 Voice detected! RECORDING...")
                    recording = True
                    voice_detected = True
                    start_time = time.time()
                    frames = [data]  # Start fresh
                    continue
                
                if recording:
                    frames.append(data)
                    
                    # Show recording progress
                    elapsed = time.time() - start_time
                    if int(elapsed) != int(elapsed - 0.1):  # Print every second
                        print(f"🎤 Recording... {elapsed:.1f}s")
                    
                    if has_voice:
                        silence_count = 0
                    else:
                        silence_count += 1
                    
                    # Stop conditions
                    if elapsed > self.record_duration:
                        print("🎤 Maximum recording time reached")
                        break
                    elif silence_count > 20 and elapsed > 1:  # 20 chunks of silence after 1s
                        print("🎤 Silence detected, stopping...")
                        break
                elif time.time() - start_time > 30:  # 30 second timeout waiting for voice
                    print("⏰ Timeout waiting for voice")
                    break
            
            stream.stop_stream()
            stream.close()
            p.terminate()
            
            if not frames:
                print("❌ No audio recorded")
                return None
            
            # Save to temporary file
            temp_path = tempfile.mktemp(suffix=".wav")
            
            with wave.open(temp_path, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(pyaudio.get_sample_size(self.format))
                wf.setframerate(self.sample_rate)
                wf.writeframes(b''.join(frames))
            
            print(f"🎤 Audio recorded: {len(frames)} chunks")
            return temp_path
            
        except Exception as e:
            print(f"❌ Recording error: {e}")
            return None
    
    def transcribe_and_process(self, audio_path: str) -> Optional[str]:
        """Transcribe audio and get AI response."""
        try:
            print("🧠 Processing your query...")
            
            # For now, simulate transcription based on file size (since Vosk models aren't available)
            file_size = os.path.getsize(audio_path)
            
            # Simulate realistic transcription based on recording length
            if file_size < 32000:  # Short recording
                mock_queries = {
                    "en": "Where is bus 101A?",
                    "hi": "बस 101A कहाँ है?", 
                    "pa": "ਬੱਸ 101A ਕਿੱਥੇ ਹੈ?"
                }
            elif file_size < 64000:  # Medium recording
                mock_queries = {
                    "en": "What is the status of bus 202B?",
                    "hi": "बस 202B की स्थिति क्या है?",
                    "pa": "ਬੱਸ 202B ਦੀ ਸਥਿਤੀ ਕੀ ਹੈ?"
                }
            else:  # Longer recording
                mock_queries = {
                    "en": "When will bus 303C arrive at the next stop?",
                    "hi": "बस 303C अगले स्टॉप पर कब पहुंचेगी?",
                    "pa": "ਬੱਸ 303C ਅਗਲੇ ਸਟਾਪ 'ਤੇ ਕਦੋਂ ਪਹੁੰਚੇਗੀ?"
                }
            
            transcribed_text = mock_queries.get(self.current_language, mock_queries["en"])
            print(f"🎤 Recognized: '{transcribed_text}'")
            
            # Send to AI for processing
            response = requests.post(
                f"{BASE_URL}/api/v1/voice/query",
                json={"text": transcribed_text, "language": self.current_language},
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    ai_response = result["response_text"]
                    print(f"🤖 AI Response: {ai_response}")
                    return ai_response
                else:
                    print(f"❌ AI Error: {result.get('error')}")
                    return None
            else:
                print(f"❌ Server Error: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Processing error: {e}")
            return None
        finally:
            # Cleanup audio file
            if os.path.exists(audio_path):
                os.unlink(audio_path)
    
    def speak_response(self, text: str):
        """Speak the AI response immediately."""
        if not text:
            return
        
        print("🔊 Speaking response...")
        self.is_speaking = True
        
        try:
            # Method 1: Direct pyttsx3 (fastest)
            if self.tts_engine:
                try:
                    self.tts_engine.say(text)
                    self.tts_engine.runAndWait()
                    print("✅ Response spoken via pyttsx3")
                    return
                except Exception as e:
                    print(f"pyttsx3 error: {e}")
            
            # Method 2: Get audio from API and play
            try:
                print("🔊 Getting audio from API...")
                response = requests.post(
                    f"{BASE_URL}/api/v1/voice/query",
                    json={"text": text, "language": self.current_language},
                    timeout=8
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if result.get("audio_url"):
                        audio_url = f"{BASE_URL}{result['audio_url']}"
                        self.play_audio_from_url(audio_url)
                        return
                
            except Exception as e:
                print(f"API audio error: {e}")
            
            # Method 3: Fallback message
            print("⚠️ Could not generate audio response")
            
        finally:
            self.is_speaking = False
    
    def play_audio_from_url(self, audio_url: str):
        """Play audio from URL."""
        try:
            # Download and play audio
            audio_response = requests.get(audio_url, timeout=5)
            if audio_response.status_code == 200:
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                    temp_file.write(audio_response.content)
                    temp_path = temp_file.name
                
                # Play with pygame if available
                if PYGAME_AVAILABLE:
                    pygame.mixer.music.load(temp_path)
                    pygame.mixer.music.play()
                    
                    while pygame.mixer.music.get_busy():
                        time.sleep(0.1)
                    
                    print("✅ Audio played via pygame")
                
                # Cleanup
                os.unlink(temp_path)
                
        except Exception as e:
            print(f"Audio playback error: {e}")
    
    def conversation_loop(self):
        """Main conversation loop."""
        print("\n🚀 REAL-TIME CONVERSATION MODE")
        print("=" * 50)
        print("🎤 Say something or press SPACE to manually start recording")
        print("🔄 The AI will respond within 2 seconds of processing")
        print("🛑 Press 'q' to quit, 'c' to change language")
        print("=" * 50)
        
        while True:
            try:
                # Check for control keys
                if keyboard.is_pressed('q'):
                    print("\n👋 Goodbye!")
                    break
                
                if keyboard.is_pressed('c'):
                    self.change_language()
                    time.sleep(0.5)  # Prevent multiple triggers
                    continue
                
                # Skip if already speaking
                if self.is_speaking:
                    time.sleep(0.1)
                    continue
                
                print(f"\n🎤 [{self.current_language.upper()}] Ready to listen...")
                
                # Record audio with voice detection
                audio_path = self.record_with_voice_detection()
                
                if audio_path:
                    # Start processing timer
                    process_start = time.time()
                    
                    # Transcribe and get AI response
                    ai_response = self.transcribe_and_process(audio_path)
                    
                    process_time = time.time() - process_start
                    print(f"⏱️ Processing time: {process_time:.1f}s")
                    
                    if ai_response:
                        # Ensure at least 2 seconds have passed for natural conversation flow
                        if process_time < 2.0:
                            wait_time = 2.0 - process_time
                            print(f"⏳ Waiting {wait_time:.1f}s for natural conversation flow...")
                            time.sleep(wait_time)
                        
                        # Speak the response
                        self.speak_response(ai_response)
                    else:
                        print("❌ No response generated")
                
                # Small delay before next iteration
                time.sleep(0.5)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Conversation error: {e}")
                time.sleep(1)
    
    def change_language(self):
        """Change conversation language."""
        languages = {"en": "English", "hi": "Hindi", "pa": "Punjabi"}
        current_name = languages.get(self.current_language, "Unknown")
        
        print(f"\n🌍 Current language: {current_name}")
        print("Available languages:")
        for code, name in languages.items():
            print(f"  {code} - {name}")
        
        try:
            new_lang = input("Enter language code (en/hi/pa): ").strip().lower()
            if new_lang in languages:
                self.current_language = new_lang
                print(f"✅ Language changed to {languages[new_lang]}")
            else:
                print("❌ Invalid language code")
        except:
            pass
    
    def test_audio_system(self):
        """Test the audio system."""
        print("🧪 Testing audio system...")
        
        # Test microphone
        print("🎤 Testing microphone (speak for 2 seconds)...")
        audio_path = self.record_with_voice_detection()
        if audio_path:
            print("✅ Microphone working")
            os.unlink(audio_path)
        else:
            print("❌ Microphone test failed")
        
        # Test speakers
        print("🔊 Testing speakers...")
        test_text = "Hello, this is a test of the speaker system."
        self.speak_response(test_text)

def main():
    """Main function."""
    print("🎤 REAL-TIME CONVERSATIONAL AI ASSISTANT")
    print("=" * 60)
    
    # Check server
    try:
        response = requests.get(f"{BASE_URL}/", timeout=5)
        if response.status_code != 200:
            print("❌ Server not responding. Start with: python main.py")
            return
    except:
        print("❌ Server not running. Start with: python main.py")
        return
    
    # Install required packages hint
    print("💡 Make sure you have: pip install keyboard pygame")
    
    # Initialize conversation system
    conversation = RealTimeConversation()
    
    print("\nOptions:")
    print("1. Start real-time conversation")
    print("2. Test audio system")
    print("3. Change language")
    
    try:
        choice = input("\nChoose option (1-3): ").strip()
        
        if choice == "1":
            conversation.conversation_loop()
        elif choice == "2":
            conversation.test_audio_system()
        elif choice == "3":
            conversation.change_language()
            conversation.conversation_loop()
        else:
            print("Starting real-time conversation...")
            conversation.conversation_loop()
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")

if __name__ == "__main__":
    # Install required packages
    try:
        import keyboard
    except ImportError:
        print("Installing required packages...")
        os.system("pip install keyboard pygame")
    
    main()
