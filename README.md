# Industrial MMS TTS System: AI-Powered Multilingual Text-to-Speech

[![Status](https://img.shields.io/badge/Status-Under%20Development-orange.svg)]()
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Gradio](https://img.shields.io/badge/Gradio-4.0+-orange.svg)](https://gradio.app/)
[![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co/transformers/)
[![Facebook](https://img.shields.io/badge/Facebook-MMS-blue.svg)](https://github.com/facebookresearch/fairseq/tree/main/examples/mms)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🚧 Repository Status

> **⚠️ UNDER DEVELOPMENT**: This repository contains an experimental Industrial Text-to-Speech System currently in active development. Features and implementation are subject to change.

**Current Phase**: Core functionality implementation and testing

---

## Technology Stack

### Core AI Frameworks
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/🤗_Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)

### Voice & Audio Processing
![Facebook](https://img.shields.io/badge/Facebook_MMS-1877F2?style=for-the-badge&logo=facebook&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)

### User Interface
![Gradio](https://img.shields.io/badge/Gradio-FF6B00?style=for-the-badge&logo=gradio&logoColor=white)
![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white)
![CSS3](https://img.shields.io/badge/CSS3-1572B6?style=for-the-badge&logo=css3&logoColor=white)

---

## Project Overview

This system aims to address communication challenges in multilingual industrial environments through AI-powered voice synthesis. The current implementation focuses on Hindi and English language support with intelligent text processing capabilities.

### Current Features

- **Bilingual Support**: Hindi and English text-to-speech conversion
- **Number Processing**: Basic conversion of digits, currency, and time formats to spoken words
- **Language Detection**: Automatic identification of Hindi vs English text
- **Message Classification**: Categorization of safety, urgent, instruction, and information messages
- **Voice Effects**: Audio modifications based on message type
- **Web Interface**: Dark-themed Gradio interface for user interaction

### Technical Implementation

The system integrates multiple AI models in a processing pipeline:

| Component | Model | Purpose |
|-----------|-------|---------|
| **TTS Engine** | `facebook/mms-tts-hin`, `facebook/mms-tts-eng` | Voice synthesis |
| **Language Detection** | `papluca/xlm-roberta-base-language-detection` | Text language identification |
| **Text Processing** | `microsoft/DialoGPT-medium` | Number-to-words conversion |
| **Classification** | `facebook/bart-large-mnli` | Message type categorization |

---

## Installation

### System Requirements
- Python 3.8 or higher
- 4GB+ RAM for model loading
- Internet connection for initial model download

### Setup Process
```bash
# Clone repository
git clone https://github.com/your-username/Industrial-MMS-TTS-System.git
cd Industrial-MMS-TTS-System

# Install dependencies
pip install -r requirements.txt

# Download AI models
python scripts/setup_models.py

# Run application
python src/main.py
```

### Access
Open browser and navigate to: `http://localhost:7863`

---

## Current Capabilities

### Text Processing
- Converts basic numbers to words (e.g., "1234" → "one thousand two hundred thirty four")
- Handles currency formats (e.g., "₹50,000" → "fifty thousand rupees")
- Processes time formats (e.g., "12:30" → "twelve thirty")

### Voice Generation
- Hindi and English speech synthesis using Facebook MMS models
- Volume adjustments based on message classification
- Repetition for safety and urgent messages

### User Interface
- Clean, dark-themed web interface
- Real-time processing feedback
- Sample text library for testing
- Auto-detection with manual override options

---

## Development Status

### Completed Components
- ✅ Basic TTS functionality for Hindi and English
- ✅ HuggingFace model integration
- ✅ Number-to-words conversion framework
- ✅ Web interface with Gradio
- ✅ Voice effect implementation

### In Progress
- [ ] Improved number conversion accuracy
- [ ] Better mixed-language text handling
- [ ] Performance optimization
- [ ] Error handling improvements

### Planned Features
- [ ] Additional Indian languages support
- [ ] Enhanced voice effects
- [ ] API development
- [ ] Mobile interface optimization

---

## Known Limitations

### Current Scope
- Limited to Hindi and English languages
- Processing time varies (2-4 seconds per request)
- Number conversion works best with common formats
- Memory usage requires 2GB+ RAM for optimal performance

### Technical Constraints
- Large model files require significant disk space
- Initial model loading takes time
- Best performance on multi-core processors
- Internet required for first-time setup

---

## Contact

**Developer**: Avinash Kumar Sharma  
**Email**: avics2020@gmail.com  


---

## License

MIT License - This experimental project is shared for educational and research purposes.

### Acknowledgments
- Facebook AI Research for MMS models
- HuggingFace team for transformer infrastructure
- Gradio team for web interface framework
- Open source community for supporting libraries

---

**Disclaimer**: This is an experimental project under active development. Features, performance, and implementation details may change significantly during development.

**Last Updated**: January 2025
