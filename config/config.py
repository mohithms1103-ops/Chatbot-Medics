import os
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any

@dataclass
class Config:
    # API Keys
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "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")
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    GOOGLE_API_KEY: Optional[str] = os.getenv("GOOGLE_API_KEY",)
    BRAVE_API_KEY: Optional[str] = os.getenv("BRAVE_API_KEY")  # For web search
    
    # LLM Settings
    DEFAULT_MODEL: str = "llama-3.1-8b-instant"
    TEMPERATURE: float = 0.7
    MAX_TOKENS: int = 1024
    TOP_P: float = 1.0
    
    # Embedding Settings
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384
    
    # RAG Settings
    VECTOR_DB_PATH: str = "medical_vectordb11.pkl"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    TOP_K_RESULTS: int = 5
    SIMILARITY_THRESHOLD: float = 0.7
    
    # Voice Settings
    TTS_RATE: int = 150
    TTS_VOLUME: float = 0.9
    SPEECH_TIMEOUT: int = 5
    PHRASE_TIME_LIMIT: int = 10
    
    # Web Search Settings
    SEARCH_ENABLED: bool = True
    MAX_SEARCH_RESULTS: int = 3
    SEARCH_TIMEOUT: int = 10
    
    # Response Settings - Fixed to use default_factory
    RESPONSE_MODES: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "concise": {
            "max_tokens": 512,
            "temperature": 0.5,
            "description": "Short, summarized replies"
        },
        "detailed": {
            "max_tokens": 1536,
            "temperature": 0.7,
            "description": "Expanded, in-depth responses"
        }
    })
    
    # File Settings
    UPLOAD_DIR: str = "uploads"
    SUPPORTED_FORMATS: List[str] = field(default_factory=lambda: [".pdf", ".txt", ".docx"])
    MAX_FILE_SIZE: int = 100 * 1024 * 1024  # 100MB
    
    # System Settings
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

# Create config instance
config = Config()

# Ensure upload directory exists
os.makedirs(config.UPLOAD_DIR, exist_ok=True)