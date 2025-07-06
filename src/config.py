"""
Configuration settings for Industrial MMS TTS System
"""

# Supported languages
SUPPORTED_LANGUAGES = {
    'english': 'facebook/mms-tts-eng',
    'hindi': 'facebook/mms-tts-hin'
}

# Server settings
DEFAULT_PORT = 7863
DEFAULT_HOST = "0.0.0.0"

# Audio settings
AUDIO_SAMPLE_RATE = 22050

# Voice effect settings
VOICE_EFFECTS = {
    'safety': {'volume_boost': 1.4, 'repetition': True},
    'urgent': {'volume_boost': 1.5, 'repetition': True},
    'instruction': {'volume_boost': 1.2, 'repetition': False},
    'information': {'volume_boost': 1.0, 'repetition': False}
}
