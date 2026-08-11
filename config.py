# config.py
# Configuration and constants

# Groq API Configuration
GROQ_API_KEY = "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"
#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
SELECTED_MODEL = "llama-3.3-70b-versatile"

# Available Groq models
AVAILABLE_MODELS = {
    "llama-3.1-70b-versatile": "llama-3.1-70b-versatile",
    "llama-3.1-8b-instant": "llama-3.1-8b-instant", 
    "mixtral-8x7b-32768": "mixtral-8x7b-32768",
    "gemma2-9b-it": "gemma2-9b-it"
}

# Default domains (fallback)
DEFAULT_DOMAINS = {
    "Machine Learning": "cs.LG",
    "Deep Learning": "cs.CV",
    "Natural Language Processing": "cs.CL",
    "Computer Vision": "cs.CV", 
    "Robotics": "cs.RO",
    "Artificial Intelligence": "cs.AI"
}

# Analysis parameters
START_YEAR = 2020
END_YEAR = 2025
NUM_TOPICS = 5
NUM_KEYWORDS = 10