import os
from dotenv import load_dotenv
import logging
from typing import Optional, Dict, Any

load_dotenv()

# ===== TOKENIZERS CONFIGURATION =====
# Disable HuggingFace tokenizers parallelism to avoid fork warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ===== DIRECTORY CONFIGURATION =====
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw")

# ===== QDRANT CONFIGURATION =====
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "collection_multimodal_rag"

# ===== GROQ API AND LLM MODELS =====
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# General LLM Model (for generic RAG queries)
LLM_MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"

# Specialized models
IMG_DESC_MODEL_LG = "meta-llama/llama-4-maverick-17b-128e-instruct"
TABLE_SUMMARY_MODEL_LG = "llama-3.3-70b-versatile"
TEXT_SUMMARY_MODEL_LG = "llama-3.3-70b-versatile"
TEXT_REWRITE_MODEL_LG = "llama-3.3-70b-versatile"

# ===== EMBEDDING CONFIGURATION =====
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/clip-ViT-B-32-multilingual-v1"
DEFAULT_BATCH_SIZE = 32
FALLBACK_TEXT_FOR_EMPTY_DOC = " "

# Score thresholds più alti per essere più selettivi
SCORE_THRESHOLD_TEXT = 0.70      # Aumentato da 0.65 a 0.70
SCORE_THRESHOLD_IMAGES = 0.75    # Aumentato da 0.70 a 0.75  
SCORE_THRESHOLD_TABLES = 0.65    # Aumentato da 0.60 a 0.65
SCORE_THRESHOLD_MIXED = 0.70     # Aumentato da 0.65 a 0.70

# Parametri adattivi - OTTIMIZZATI per massimo 7 risultati totali
ADAPTIVE_K_MIN = 2               # Minimo risultati da restituire
ADAPTIVE_K_MAX = 4               # Massimo risultati per tipo (4*3 = 12, ma con balancing = max 7)

# Strategies for different query types - K OTTIMIZZATI PER 7 RISULTATI TOTALI
RAG_PARAMS: Dict[str, Dict[str, Any]] = {
    "factual": {
        "k": 3,                  # Ridotto per essere precisi
        "score_threshold": 0.60, 
        "description": "For precise factual questions"
    },
    "exploratory": {
        "k": 4,                  # Leggermente più alto per esplorazione
        "score_threshold": 0.55, 
        "description": "For broad exploratory searches"
    },
    "technical": {
        "k": 3,                  # Preciso per domande tecniche
        "score_threshold": 0.65, 
        "description": "For specific technical queries"
    },
    "multimodal": {
        "k": 3,                  # Bilanciato per multimodal
        "score_threshold": 0.50, 
        "description": "For searches including text, images and tables"
    }
}

# global limit to prevent too many results
MAX_GLOBAL_DOCUMENTS = 7

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = os.path.join(PROJECT_ROOT, "logs", "multimodal_rag.log")

# Performance Settings
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "10"))
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "3600"))

# Function to validate configuration
def validate_config() -> Optional[str]:
    """Validates the configuration settings and returns an error message if any required setting is missing."""
    if not GROQ_API_KEY:
        return "GROQ_API_KEY is required but not configured"

    if not os.path.exists(os.path.dirname(LOG_FILE)):
        try:
            os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        except Exception as e:
            return f"Unable to create log directory: {e}"

    return None

# Setup logging
def setup_logging() -> None:
    """Configures the application logging"""
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL.upper()),
        format=LOG_FORMAT,
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler()
        ]
    )
    # Set specific levels for external libraries
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("qdrant_client").setLevel(logging.INFO)