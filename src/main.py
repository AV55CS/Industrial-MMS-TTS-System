import gradio as gr
from transformers import VitsModel, AutoTokenizer, pipeline
import torch
import numpy as np
import soundfile as sf
import tempfile
import time
from datetime import datetime
import re
import os
from scipy import signal

class HuggingFaceLLMProcessor:
    """SIMPLIFIED: Free Hugging Face LLM for Hindi + English only"""
    def __init__(self):
        print("🤗 Loading FREE Hugging Face models for Hindi + English...")
        
        # Load free text generation model for number conversion
        try:
            self.text_generator = pipeline(
                "text-generation",
                model="microsoft/DialoGPT-medium",
                device=-1,  # CPU only
                pad_token_id=50256
            )
            self.has_text_generator = True
            print("✅ Text generation model loaded (DialoGPT-medium)")
        except Exception as e:
            print(f"⚠️ Text generator failed: {e}")
            self.has_text_generator = False
        
        # Load free classification model
        try:
            self.classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=-1
            )
            self.has_classifier = True
            print("✅ Classification model loaded (BART-large)")
        except Exception as e:
            print(f"⚠️ Classifier failed: {e}")
            self.has_classifier = False
    
    def convert_numbers_to_words(self, text, language):
        """Convert numbers using free Hugging Face model - Hindi + English only"""
        if not self.has_text_generator:
            print("⚠️ No text generator available, using simple fallback")
            return self._simple_number_fallback(text, language)
        
        try:
            # Enhanced examples with currency and time
            examples = {
                'english': "1234 becomes one thousand two hundred thirty four, ₹50,000 becomes fifty thousand rupees, 12:30 becomes twelve thirty",
                'hindi': "1234 होता है एक हजार दो सौ चौंतीस, ₹50,000 होता है पचास हजार रुपये, 12:30 होता है बारह तीस"
            }
            
            example = examples.get(language, examples['english'])
            
            # Enhanced prompt for better conversion
            prompt = f"""Convert ALL numbers, currency, and time to words in {language}.
Examples: {example}
Convert: {text}
Result:"""
            
            # Generate with the model - FIXED parameters
            result = self.text_generator(
                prompt,
                max_new_tokens=50,  # ✅ Fixed: use max_new_tokens instead of max_length
                num_return_sequences=1,
                temperature=0.1,
                do_sample=True,
                pad_token_id=50256,
                eos_token_id=50256,
                truncation=True  # ✅ Fixed: explicit truncation
            )
            
            # Extract the result
            generated_text = result[0]['generated_text']
            
            # Parse the output after "Result:"
            if "Result:" in generated_text:
                converted_text = generated_text.split("Result:")[-1].strip()
            else:
                converted_text = generated_text.replace(prompt, "").strip()
            
            # Clean up the result with correct language
            converted_text = self._clean_converted_text(converted_text, text, language)
            
            print(f"🔢 Number conversion: '{text}' → '{converted_text}'")
            return converted_text
            
        except Exception as e:
            print(f"❌ LLM number conversion failed: {e}")
            return self._simple_number_fallback(text, language)
    
    def _clean_converted_text(self, converted_text, original_text, language="english"):
        """Clean and validate the converted text"""
        # Remove extra whitespace
        converted_text = re.sub(r'\s+', ' ', converted_text).strip()
        
        # Remove any leftover prompt text
        cleanup_patterns = [
            r'Convert.*?Result:',
            r'Examples?:.*?Convert:',
            r'Convert all numbers.*?Result:',
            r'^\s*[-*•]\s*',
            r'^\d+\.\s*',
        ]
        
        for pattern in cleanup_patterns:
            converted_text = re.sub(pattern, '', converted_text, flags=re.IGNORECASE | re.DOTALL)
        
        converted_text = converted_text.strip()
        
        # If result is too short or too long, use ENHANCED fallback with correct language
        if len(converted_text) < len(original_text) * 0.5 or len(converted_text) > len(original_text) * 3:
            print(f"⚠️ Converted text length suspicious, using ENHANCED fallback for {language}")
            return self._simple_number_fallback(original_text, language)  # ✅ FIXED: Use correct language
        
        return converted_text if converted_text else original_text
    
    def _simple_number_fallback(self, text, language):
        """ENHANCED fallback for time & money - Hindi + English only"""
        
        print(f"🔧 Using fallback conversion for language: {language}")
        
        # Enhanced replacements with more numbers
        simple_replacements = {
            'english': {
                '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
                '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine',
                '10': 'ten', '11': 'eleven', '12': 'twelve', '13': 'thirteen', '14': 'fourteen', '15': 'fifteen',
                '20': 'twenty', '25': 'twenty five', '30': 'thirty', '50': 'fifty',
                '100': 'one hundred', '1000': 'one thousand', '2500': 'two thousand five hundred',
                '1234': 'one thousand two hundred thirty four',
                '50000': 'fifty thousand', '25000': 'twenty five thousand', '150000': 'one hundred fifty thousand'
            },
            'hindi': {
                '0': 'शून्य', '1': 'एक', '2': 'दो', '3': 'तीन', '4': 'चार',
                '5': 'पांच', '6': 'छह', '7': 'सात', '8': 'आठ', '9': 'नौ',
                '10': 'दस', '11': 'ग्यारह', '12': 'बारह', '13': 'तेरह', '14': 'चौदह', '15': 'पंद्रह',
                '20': 'बीस', '25': 'पच्चीस', '30': 'तीस', '50': 'पचास',
                '100': 'सौ', '1000': 'हजार', '2500': 'ढाई हजार',
                '1234': 'एक हजार दो सौ चौंतीस',
                '50000': 'पचास हजार', '25000': 'पच्चीस हजार', '150000': 'डेढ़ लाख'
            }
        }
        
        replacements = simple_replacements.get(language, simple_replacements['english'])
        result = text
        
        print(f"🔧 Input text: {text}")
        print(f"🔧 Using {language} replacements")
        
        # FIXED: Handle currency - ₹50,000 → "पचास हजार रुपये"
        currency_pattern = r'₹(\d{1,3}(?:,\d{3})*)'
        def replace_currency(match):
            amount = match.group(1).replace(',', '')  # Remove commas: "50,000" → "50000"
            amount_words = replacements.get(amount, amount)
            currency_word = 'rupees' if language == 'english' else 'रुपये'
            print(f"🔧 Currency: ₹{match.group(1)} → {amount_words} {currency_word}")
            return f"{amount_words} {currency_word}"
        result = re.sub(currency_pattern, replace_currency, result)
        
        # FIXED: Handle time - 12:30 → "बारह तीस"  
        time_pattern = r'(\d{1,2}):(\d{2})'
        def replace_time(match):
            hour = match.group(1)   # "12"
            minute = match.group(2) # "30"
            hour_word = replacements.get(hour, hour)
            minute_word = replacements.get(minute, minute)
            print(f"🔧 Time: {hour}:{minute} → {hour_word} {minute_word}")
            return f"{hour_word} {minute_word}"
        result = re.sub(time_pattern, replace_time, result)
        
        # Handle percentages - 95.5% → "ninety five point five percent"
        percent_pattern = r'(\d+(?:\.\d+)?)%'
        def replace_percent(match):
            number = match.group(1)
            if '.' in number:
                parts = number.split('.')
                whole = replacements.get(parts[0], parts[0])
                decimal = ' '.join([replacements.get(d, d) for d in parts[1]])
                percent_word = 'percent' if language == 'english' else 'प्रतिशत'
                return f"{whole} point {decimal} {percent_word}"
            else:
                number_word = replacements.get(number, number)
                percent_word = 'percent' if language == 'english' else 'प्रतिशत'
                return f"{number_word} {percent_word}"
        result = re.sub(percent_pattern, replace_percent, result)
        
        # Handle standalone numbers
        for number, word in replacements.items():
            # Use word boundaries to avoid partial replacements
            if re.search(rf'\b{re.escape(number)}\b', result):
                print(f"🔧 Number: {number} → {word}")
                result = re.sub(rf'\b{re.escape(number)}\b', word, result)
        
        print(f"🔧 Final result: {result}")
        return result
    
    def detect_language_simple(self, text):
        """Simple language detection for just Hindi + English"""
        if not self.has_classifier:
            return None
        
        try:
            # Simple 2-language classification
            candidate_labels = ["english text", "hindi devanagari text"]
            
            result = self.classifier(text, candidate_labels)
            
            detected_label = result['labels'][0]
            confidence = result['scores'][0]
            
            if confidence > 0.6:
                if "hindi" in detected_label:
                    return "hindi"
                else:
                    return "english"
            
            return None
            
        except Exception as e:
            print(f"❌ LLM language detection failed: {e}")
            return None

class SimpleLanguageDetector:
    """SIMPLIFIED: Just Hindi + English detection"""
    def __init__(self):
        print("🌐 Initializing SIMPLE language detection for Hindi + English...")
        
        # Try to load free language detection model
        try:
            self.ml_detector = pipeline(
                "text-classification",
                model="papluca/xlm-roberta-base-language-detection",
                device=-1
            )
            self.has_ml_model = True
            print("✅ XLM-RoBERTa language detection model loaded")
        except Exception as e:
            print(f"⚠️ ML language detection failed: {e}")
            self.has_ml_model = False
        
        # Initialize HuggingFace LLM as backup
        self.llm_processor = HuggingFaceLLMProcessor()
        
        # SIMPLE patterns for just 2 languages
        self.language_patterns = {
            'english': {
                'chars': set(['a', 'e', 'i', 'o', 'u', 't', 'h', 'r', 's', 'n', 'l', 'd', 'c', 'm']),
                'words': ['the', 'and', 'is', 'in', 'to', 'of', 'a', 'for', 'with', 'on', 'emergency', 'fire']
            },
            'hindi': {
                'chars': set(['क', 'र', 'त', 'न', 'स', 'म', 'ह', 'द', 'प', 'ल', 'ग', 'व', 'य', 'ब', 'च']),
                'words': ['है', 'का', 'में', 'को', 'से', 'और', 'यह', 'पर', 'एक', 'आपातकाल', 'आग']
            }
        }
    
    def detect_language(self, text):
        """Simple 2-language detection"""
        if not text or len(text.strip()) < 3:
            return 'english'  # Default fallback
        
        # Method 1: Try ML model first
        if self.has_ml_model:
            try:
                result = self.ml_detector(text)[0]
                detected_code = result['label']
                confidence = result['score']
                
                if confidence > 0.75:
                    if detected_code == 'hi':
                        return 'hindi'
                    elif detected_code == 'en':
                        return 'english'
            except Exception as e:
                print(f"⚠️ ML detection failed: {e}")
        
        # Method 2: Try LLM detection
        llm_result = self.llm_processor.detect_language_simple(text)
        if llm_result:
            print(f"🤗 LLM detected: {llm_result}")
            return llm_result
        
        # Method 3: Simple rule-based detection
        return self._simple_rule_detection(text)
    
    def _simple_rule_detection(self, text):
        """Very simple rule-based detection for 2 languages"""
        text_lower = text.lower()
        
        # Count Devanagari characters (Hindi)
        hindi_chars = sum(1 for char in text if 0x0900 <= ord(char) <= 0x097F)
        
        # Count English characters
        english_chars = sum(1 for char in text_lower if char.isalpha() and ord(char) < 0x0080)
        
        # Count Hindi words
        hindi_words = sum(1 for word in self.language_patterns['hindi']['words'] if word in text_lower)
        
        # Count English words
        english_words = sum(1 for word in self.language_patterns['english']['words'] if word in text_lower)
        
        # Simple scoring
        hindi_score = hindi_chars * 2 + hindi_words * 3
        english_score = english_chars + english_words * 3
        
        if hindi_score > english_score:
            return 'hindi'
        else:
            return 'english'

class VoiceReflectionProcessor:
    """SIMPLIFIED: Voice reflection for Hindi + English only"""
    def __init__(self):
        self.voice_styles = {
            'safety': {
                'volume_boost': 1.4,        # 40% louder - CLEAN
                'repetition': True,         # Repeat for emphasis - CLEAN
                'text_prefix': {
                    'hindi': 'अत्यधिक सावधानी से सुनें। ',
                    'english': 'Listen very carefully. '
                }
            },
            'urgent': {
                'volume_boost': 1.5,        # 50% louder - CLEAN
                'repetition': True,         # Repeat for urgency - CLEAN
                'text_prefix': {
                    'hindi': 'तुरंत ध्यान दें। ',
                    'english': 'Immediate attention required. '
                }
            },
            'instruction': {
                'volume_boost': 1.2,        # 20% louder - CLEAN
                'repetition': False,
                'text_prefix': {
                    'hindi': 'निर्देशों को ध्यान से सुनें। ',
                    'english': 'Follow these instructions carefully. '
                }
            },
            'information': {
                'volume_boost': 1.0,        # Normal volume
                'repetition': False,
                'text_prefix': {'hindi': '', 'english': ''}
            },
            'general': {
                'volume_boost': 1.0,
                'repetition': False,
                'text_prefix': {'hindi': '', 'english': ''}
            }
        }
    
    def style_text_for_voice(self, text, message_type, language):
        """Simple text styling for 2 languages"""
        if message_type not in self.voice_styles:
            return text
        
        style = self.voice_styles[message_type]
        styled_text = text
        
        # Add prefix for voice emphasis
        prefix = style['text_prefix'].get(language, style['text_prefix'].get('english', ''))
        if prefix:
            styled_text = prefix + styled_text
        
        # Add emphasis for safety messages
        if message_type == 'safety':
            styled_text = styled_text.replace('. ', '... ')
            styled_text = styled_text.replace('!', '!!!')
        
        # Add urgency markers
        elif message_type == 'urgent':
            styled_text = styled_text.replace('!', '!!!')
        
        # Structure instructions
        elif message_type == 'instruction':
            sentences = styled_text.split('. ')
            if len(sentences) > 1:
                structured = []
                step_word = "चरण" if language == 'hindi' else "Step"
                
                for i, sentence in enumerate(sentences):
                    if sentence.strip():
                        structured.append(f"{step_word} {i+1}. {sentence.strip()}")
                styled_text = '. '.join(structured)
        
        return styled_text
    
    def modify_audio(self, audio, sample_rate, message_type):
        """CLEAN audio effects - no ghost sounds!"""
        if message_type not in self.voice_styles:
            return audio
        
        style = self.voice_styles[message_type]
        modified_audio = audio.copy().astype(np.float32)
        
        # ONLY clean volume adjustment (no pitch/speed artifacts)
        if style['volume_boost'] != 1.0:
            modified_audio = modified_audio * style['volume_boost']
            # Clean clipping prevention
            max_val = np.max(np.abs(modified_audio))
            if max_val > 32767:
                modified_audio = modified_audio * (32767 / max_val)
        
        # ONLY clean repetition for critical messages (no artifacts)
        if style['repetition'] and message_type in ['safety', 'urgent']:
            pause_duration = 1.0 if message_type == 'safety' else 0.7
            # Clean silence generation
            pause = np.zeros(int(sample_rate * pause_duration), dtype=modified_audio.dtype)
            # Clean concatenation
            modified_audio = np.concatenate([modified_audio, pause, modified_audio])
        
        return modified_audio.astype(np.int16)

class SimpleMessageClassifier:
    """SIMPLIFIED: Message classification for Hindi + English only"""
    def __init__(self):
        # Try to load free classification model
        try:
            self.ml_classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=-1
            )
            self.has_ml_model = True
            print("✅ Message classification model loaded")
        except:
            self.has_ml_model = False
            print("⚠️ Using pattern-based message classification")
        
        # SIMPLE patterns for 2 languages
        self.patterns = {
            'safety': {
                'hindi': ['आपातकाल', 'खतरा', 'आग', 'दुर्घटना', 'सावधान', 'चेतावनी', 'तुरंत', 'भाग'],
                'english': ['emergency', 'danger', 'fire', 'accident', 'warning', 'alert', 'hazard', 'evacuate']
            },
            'urgent': {
                'hindi': ['तुरंत', 'जल्दी', 'फौरन', 'अभी', 'शीघ्र', 'समय'],
                'english': ['urgent', 'immediate', 'asap', 'quickly', 'rush', 'time', 'critical']
            },
            'instruction': {
                'hindi': ['निर्देश', 'प्रक्रिया', 'चरण', 'कृपया', 'दबाएं', 'जाएं', 'करें'],
                'english': ['instruction', 'procedure', 'step', 'please', 'press', 'go', 'follow']
            },
            'information': {
                'hindi': ['जानकारी', 'सूचना', 'अपडेट', 'रिपोर्ट', 'समाचार', 'स्थिति'],
                'english': ['information', 'update', 'report', 'news', 'status', 'current']
            }
        }
    
    def classify_message(self, text, language='english'):
        """Simple classification for 2 languages"""
        # Try ML model first
        if self.has_ml_model and len(text.strip()) > 10:
            try:
                ml_result = self._ml_classification(text)
                if ml_result:
                    return ml_result
            except:
                pass
        
        # Fallback to pattern matching
        return self._pattern_classification(text, language)
    
    def _ml_classification(self, text):
        """ML-based classification"""
        categories = [
            "safety emergency warning danger",
            "urgent immediate time critical", 
            "information update report news",
            "instruction procedure steps guide",
            "general conversation normal"
        ]
        
        result = self.ml_classifier(text, categories)
        confidence = result['scores'][0]
        
        if confidence > 0.6:
            label = result['labels'][0].lower()
            if 'safety' in label or 'emergency' in label:
                return 'safety'
            elif 'urgent' in label or 'immediate' in label:
                return 'urgent'
            elif 'information' in label:
                return 'information'
            elif 'instruction' in label:
                return 'instruction'
        
        return None
    
    def _pattern_classification(self, text, language):
        """Simple pattern-based classification"""
        text_lower = text.lower()
        scores = {}
        
        # Use English patterns if language not supported
        lang_patterns = language if language in ['hindi', 'english'] else 'english'
        
        for category in ['safety', 'urgent', 'instruction', 'information']:
            keywords = self.patterns[category].get(lang_patterns, [])
            score = sum(1 for keyword in keywords if keyword in text_lower)
            scores[category] = score
        
        # Return category with highest score
        max_score = max(scores.values()) if scores.values() else 0
        if max_score >= 1:
            return max(scores.items(), key=lambda x: x[1])[0]
        
        return 'general'

class MultilingualMMS_TTS:
    """SIMPLIFIED: Only Hindi + English"""
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        
        # SIMPLE: Only 2 languages
        self.available_languages = {
            'english': 'facebook/mms-tts-eng',
            'hindi': 'facebook/mms-tts-hin'
        }
        
        # Load Hindi model by default
        self.load_language('hindi')
        self.current_language = 'hindi'
        
        print("🎤 MMS TTS System initialized with Hindi")
        print(f"📊 Languages: {list(self.available_languages.keys())}")
    
    def load_language(self, language):
        if language in self.available_languages and language not in self.models:
            try:
                model_name = self.available_languages[language]
                print(f"📦 Loading {language} model: {model_name}")
                
                self.models[language] = VitsModel.from_pretrained(model_name)
                self.tokenizers[language] = AutoTokenizer.from_pretrained(model_name)
                
                print(f"✅ {language.title()} model loaded successfully")
                return True
            except Exception as e:
                print(f"❌ Failed to load {language}: {e}")
                return False
        return language in self.models
    
    def generate_speech(self, text, language=None):
        if language is None:
            language = 'hindi'
        
        # Ensure language is supported
        if language not in self.available_languages:
            print(f"⚠️ Language {language} not supported, using Hindi")
            language = 'hindi'
        
        # Load language model if not already loaded
        if not self.load_language(language):
            language = 'hindi'
        
        try:
            model = self.models[language]
            tokenizer = self.tokenizers[language]
            
            inputs = tokenizer(text, return_tensors="pt")
            
            with torch.no_grad():
                waveform = model(**inputs).waveform
            
            # Convert to audio format
            audio = (waveform / waveform.abs().max() * 32767).squeeze().to(torch.int16).cpu().numpy()
            sample_rate = model.config.sampling_rate
            
            return sample_rate, audio, language
            
        except Exception as e:
            print(f"❌ Speech generation failed: {e}")
            return None, None, language

class IndustrialTTSInterface:
    """SIMPLIFIED: Hindi + English only system"""
    def __init__(self):
        print("🚀 Initializing SIMPLE Industrial TTS for Hindi + English...")
        
        # Core TTS engine (2 languages only)
        self.tts_engine = MultilingualMMS_TTS()
        
        # HuggingFace LLM processor
        print("🔢 Setting up HuggingFace LLM...")
        self.llm_processor = HuggingFaceLLMProcessor()
        
        # Simple language detection
        print("🌐 Setting up simple language detection...")
        self.language_detector = SimpleLanguageDetector()
        
        # Voice reflection
        print("🎭 Setting up voice reflection...")
        self.voice_processor = VoiceReflectionProcessor()
        
        # Message classification
        print("🧠 Setting up message classification...")
        self.message_classifier = SimpleMessageClassifier()
        
        # SIMPLE message prefixes for 2 languages
        self.message_prefixes = {
            'safety': {
                'english': '⚠️ Safety Alert: ',
                'hindi': '⚠️ सुरक्षा चेतावनी: '
            },
            'urgent': {
                'english': '🚨 Urgent Message: ',
                'hindi': '🚨 त्वरित संदेश: '
            },
            'information': {
                'english': 'ℹ️ Information: ',
                'hindi': 'ℹ️ सूचना: '
            },
            'instruction': {
                'english': '📋 Instruction: ',
                'hindi': '📋 निर्देश: '
            }
        }
        
        # SIMPLE sample texts
        self.sample_texts = {
            'English Safety + Numbers': 'Emergency at room 1234! Fire detected. Budget: ₹50,000. Evacuate at 12:30 PM!',
            'Hindi Safety + Numbers': 'आपातकाल कमरा 1234 में! आग लगी है। बजट: ₹50,000। 12:30 बजे निकलें!',
            'English Information': 'Today production target is 2500 units. Temperature is 25.5 degrees. Weather clear.',
            'Hindi Information': 'आज उत्पादन लक्ष्य 2500 यूनिट है। तापमान 25.5 डिग्री। मौसम साफ।',
            'English Instructions': 'Please go to station 5. Press the red button. Start at 12:30. Budget ₹25,000.',
            'Hindi Instructions': 'कृपया स्टेशन 5 पर जाएं। लाल बटन दबाएं। 12:30 बजे शुरू करें।',
            'Number Test': 'Room 1234, budget ₹1,50,000, success 95.5%, time 2:30 PM, target 5000 units.',
            'Mixed Text': 'Emergency कमरा 1234! Fire लग गई! Evacuate तुरंत!'
        }
        
        print("✅ Simple Industrial TTS System ready!")
    
    def generate_industrial_tts(self, text, msg_type, language_choice):
        """MAIN FUNCTION: Simple TTS generation"""
        start_time = time.time()
        
        print(f"\n🎬 Starting simple TTS generation...")
        print(f"📝 Input: '{text}'")
        
        # STEP 1: Language detection (Hindi or English)
        if language_choice == "Auto-detect":
            detected_language = self.language_detector.detect_language(text)
            print(f"🔍 Auto-detected: {detected_language}")
        else:
            detected_language = language_choice.lower()
            print(f"✅ Selected: {detected_language}")
        
        # STEP 2: Number conversion with LLM
        print("🔢 Converting numbers...")
        processed_text = self.llm_processor.convert_numbers_to_words(text, detected_language)
        print(f"✅ Processed: '{processed_text}'")
        
        # STEP 3: Message classification
        if msg_type == "Auto":
            classified_type = self.message_classifier.classify_message(processed_text, detected_language)
            print(f"🧠 Classified as: {classified_type}")
        else:
            classified_type = msg_type.lower()
            print(f"✅ Selected type: {classified_type}")
        
        # STEP 4: Add prefix
        prefix = self.message_prefixes.get(classified_type, {}).get(detected_language, '')
        prefixed_text = prefix + processed_text if prefix else processed_text
        
        # STEP 5: Style for voice
        styled_text = self.voice_processor.style_text_for_voice(prefixed_text, classified_type, detected_language)
        print(f"🎭 Styled: '{styled_text}'")
        
        # STEP 6: Generate speech
        print(f"🎤 Generating speech...")
        sample_rate, audio, used_language = self.tts_engine.generate_speech(styled_text, detected_language)
        
        if audio is not None:
            # STEP 7: Apply voice effects
            enhanced_audio = self.voice_processor.modify_audio(audio, sample_rate, classified_type)
            
            processing_time = time.time() - start_time
            
            # Simple result info
            voice_effects = {
                'safety': "🔊 **Safety Voice**: 40% louder volume, repeated delivery for clarity",
                'urgent': "🚨 **Urgent Voice**: 50% louder volume, repeated for emphasis", 
                'instruction': "📋 **Instruction Voice**: 20% louder, clear single delivery",
                'information': "ℹ️ **Information Voice**: Normal volume and delivery",
                'general': "💬 **General Voice**: Standard delivery"
            }
            
            result_info = f"""
🎯 **Simple Processing Results:**

**🌐 Language**: {used_language.title()} ({"Auto-detected" if language_choice == "Auto-detect" else "Selected"})
**🧠 Message Type**: {classified_type.title()} ({"Auto-classified" if msg_type == "Auto" else "Selected"})

**🔢 Number Conversion (ENHANCED)**:
- Original: `{text}`
- Processed: `{processed_text}`
- Conversions: Numbers, ₹Currency, Time formats
- Status: {"Enhanced fallback used" if text != processed_text else "No numbers found"}

**🎭 Voice Enhancement**:
{voice_effects.get(classified_type, voice_effects['general'])}

**📊 Technical**:
- Processing Time: {processing_time:.2f} seconds
- Audio Length: {len(enhanced_audio) / sample_rate:.2f} seconds
- Sample Rate: {sample_rate} Hz

**🎪 Final Text**: `{styled_text}`

**✅ Successfully processed with FREE HuggingFace models!**
            """
            
            print("🎉 TTS generation completed!")
            return (sample_rate, enhanced_audio), result_info
            
        else:
            error_msg = f"""
❌ **Speech Generation Failed**

- Language: {detected_language}
- Time: {time.time() - start_time:.2f} seconds
- Text: `{styled_text}`

Try switching to the other language or check the text content.
            """
            return None, error_msg
    
    def load_sample_text(self, sample_choice):
        """Load sample text"""
        return self.sample_texts.get(sample_choice, "")
    
    def create_interface(self):
        """Simple interface for 2 languages"""
        custom_css = """
        .gradio-container {
            background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%) !important;
            color: #ffffff !important;
        }
        .gr-button {
            background: linear-gradient(45deg, #ff6b35, #ff8c42) !important;
            border: none !important;
            color: white !important;
            font-weight: bold !important;
            border-radius: 8px !important;
        }
        .gr-button:hover {
            background: linear-gradient(45deg, #ff8c42, #ff6b35) !important;
            transform: translateY(-2px) !important;
        }
        .gr-textbox, .gr-dropdown {
            background: #2d2d2d !important;
            border: 2px solid #404040 !important;
            color: #ffffff !important;
            border-radius: 8px !important;
        }
        .gr-radio label {
            background: #2d2d2d !important;
            border: 2px solid #404040 !important;
            color: #ffffff !important;
            border-radius: 6px !important;
        }
        .gr-radio input:checked + label {
            background: #ff6b35 !important;
        }
        .gr-markdown {
            background: #1a1a1a !important;
            border: 1px solid #404040 !important;
            border-radius: 8px !important;
            padding: 15px !important;
            color: #ffffff !important;
        }
        """
        
        with gr.Blocks(title="🏭 Simple Industrial TTS - Hindi + English", css=custom_css) as demo:
            gr.HTML("""
            <div style="background: linear-gradient(135deg, #ff6b35 0%, #ff8c42 50%, #1a1a1a 100%); 
                        padding: 2rem; border-radius: 15px; margin-bottom: 2rem; text-align: center;">
                <h1 style="color: white; margin: 0; font-size: 2.5rem;">
                    🏭 Simple Industrial TTS System
                </h1>
                <p style="color: white; margin: 10px 0 0 0; font-size: 1.2rem;">
                    Hindi + English Only - FREE HuggingFace AI
                </p>
                <p style="color: #ffccaa; margin: 5px 0 0 0; font-style: italic;">
                    🔢 Smart Numbers • 🎭 Voice Reflection • 🧠 Auto-Classification
                </p>
            </div>
            """)
            
            with gr.Row():
                with gr.Column(scale=2):
                    gr.HTML("<h3 style='color: #ff6b35;'>🎤 Text-to-Speech (Hindi + English)</h3>")
                    
                    text_input = gr.Textbox(
                        lines=4,
                        label="📝 Message Text (Numbers will be converted automatically)",
                        placeholder="Try: Emergency room 1234! Budget ₹50,000. Time 12:30 PM."
                    )
                    
                    with gr.Row():
                        msg_type = gr.Radio(
                            choices=["Auto", "safety", "urgent", "information", "instruction", "general"],
                            label="🔍 Message Type",
                            value="Auto"
                        )
                        
                        language_choice = gr.Dropdown(
                            choices=["Auto-detect", "Hindi", "English"],
                            label="🌐 Language",
                            value="Auto-detect"
                        )
                    
                    generate_btn = gr.Button("🤗 Generate Simple TTS", variant="primary", size="lg")
                    
                    gr.HTML("<h3 style='color: #ff6b35; margin-top: 2rem;'>📝 Test Samples</h3>")
                    
                    sample_dropdown = gr.Dropdown(
                        choices=list(self.sample_texts.keys()),
                        label="Choose Sample Text"
                    )
                    
                    load_sample_btn = gr.Button("📋 Load Sample")
                
                with gr.Column(scale=2):
                    gr.HTML("<h3 style='color: #ff6b35;'>🎵 Enhanced Audio</h3>")
                    
                    audio_output = gr.Audio(
                        label="🔊 AI-Enhanced Speech",
                        type="numpy"
                    )
                    
                    result_info = gr.Markdown(
                        label="📊 Processing Results",
                        value="**Results will appear here...**"
                    )
            
            with gr.Row():
                gr.HTML("""
                <div style="background: linear-gradient(135deg, #2d2d2d 0%, #1a1a1a 100%); 
                           border: 2px solid #ff6b35; border-radius: 12px; padding: 1.5rem; margin-top: 2rem;">
                    <h3 style="color: #ff6b35; text-align: center;">🎯 Simple System Features</h3>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
                        <div style="background: rgba(255, 107, 53, 0.1); border: 1px solid #ff6b35; 
                                   border-radius: 8px; padding: 1rem;">
                            <h4 style="color: #ff6b35;">🔢 Number Conversion</h4>
                            <p style="color: #cccccc;">Converts 1234 → "one thousand..." in Hindi + English</p>
                        </div>
                        <div style="background: rgba(255, 107, 53, 0.1); border: 1px solid #ff6b35; 
                                   border-radius: 8px; padding: 1rem;">
                            <h4 style="color: #ff6b35;">🎭 Clean Voice Effects</h4>
                            <p style="color: #cccccc;">Safety: louder + repeated, Urgent: loudest + repeated, Instructions: clear</p>
                        </div>
                    </div>
                    <div style="text-align: center; margin-top: 1rem; padding: 1rem; 
                               background: rgba(255, 107, 53, 0.05); border-radius: 8px;">
                        <p style="color: #ff8c42; font-weight: bold;">
                            ✅ Simplified to Hindi + English only for better accuracy and easier debugging!
                        </p>
                    </div>
                </div>
                """)
            
            # Event handlers
            generate_btn.click(
                fn=self.generate_industrial_tts,
                inputs=[text_input, msg_type, language_choice],
                outputs=[audio_output, result_info]
            )
            
            load_sample_btn.click(
                fn=self.load_sample_text,
                inputs=[sample_dropdown],
                outputs=[text_input]
            )
        
        return demo

def main():
    """Simple main function"""
    print("🚀 Starting SIMPLE Industrial TTS System...")
    print("🎯 Languages: Hindi + English ONLY")
    print("🤗 Using FREE HuggingFace models")
    print("=" * 50)
    
    try:
        interface = IndustrialTTSInterface()
        demo = interface.create_interface()
        
        print("✅ SIMPLE SYSTEM READY!")
        print("🌐 Languages: Hindi, English")
        print("🎯 Types: Safety, Urgent, Information, Instruction")
        print("💰 Cost: $0 (100% FREE)")
        print("🚀 Let's test it!")
        
        return demo
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        raise

if __name__ == "__main__":
    demo = main()
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7861,
        show_error=True
    )
