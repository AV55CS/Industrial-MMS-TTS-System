#!/usr/bin/env python3
"""
Model Setup Script - Downloads all required AI models
"""

from transformers import VitsModel, AutoTokenizer, pipeline

def download_models():
    print("📦 Downloading AI models for Industrial TTS...")
    
    # TTS Models
    print("1. 🎤 Hindi TTS...")
    VitsModel.from_pretrained("facebook/mms-tts-hin")
    AutoTokenizer.from_pretrained("facebook/mms-tts-hin")
    
    print("2. 🎤 English TTS...")
    VitsModel.from_pretrained("facebook/mms-tts-eng")
    AutoTokenizer.from_pretrained("facebook/mms-tts-eng")
    
    # AI Models
    print("3. 🌐 Language Detection...")
    pipeline("text-classification", model="papluca/xlm-roberta-base-language-detection")
    
    print("4. 🔢 Number Conversion...")
    pipeline("text-generation", model="microsoft/DialoGPT-medium")
    
    print("5. 🧠 Message Classification...")
    pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    
    print("✅ All models downloaded successfully!")

if __name__ == "__main__":
    download_models()
