#!/usr/bin/env python3
"""
Quick Demo Script
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def run_demo():
    print("🎬 Industrial TTS Demo")
    
    try:
        from main import IndustrialTTSInterface
        
        interface = IndustrialTTSInterface()
        
        # Test texts
        tests = [
            "Emergency at room 1234! Budget ₹50,000!",
            "आपातकाल कमरा 1234 में! बजट ₹50,000!"
        ]
        
        for text in tests:
            print(f"Testing: {text}")
            result = interface.generate_industrial_tts(text, "Auto", "Auto-detect")
            if result[0]:
                print("✅ Success!")
            else:
                print("❌ Failed")
                
    except Exception as e:
        print(f"❌ Demo failed: {e}")

if __name__ == "__main__":
    run_demo()
