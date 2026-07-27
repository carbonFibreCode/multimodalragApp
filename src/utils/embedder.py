import logging
from typing import List, Optional
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from langchain_core.embeddings import Embeddings
from src.config import DEFAULT_EMBEDDING_MODEL, DEFAULT_BATCH_SIZE, FALLBACK_TEXT_FOR_EMPTY_DOC

logger = logging.getLogger(__name__)

class AdvancedEmbedder(Embeddings):
    """Advanced embedder for text and image descriptions"""
    
    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        batch_size: int = DEFAULT_BATCH_SIZE,
        device: Optional[str] = None,
        trust_remote_code: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device or "cpu"
        
        try:
            self.model = HuggingFaceEmbedding(
                model_name=self.model_name,
                device=self.device,
                trust_remote_code=trust_remote_code,
                embed_batch_size=self.batch_size
            )
            self._determine_embedding_dim()
            logger.info(f"Embedder initialized with {model_name}")
        except Exception as e:
            logger.error(f"Initialization error: {e}")
            raise

    def _determine_embedding_dim(self):
        """Automatically determine embedding dimensions"""
        try:
            test_embedding = self.model.get_text_embedding("test")
            self.embedding_dim = len(test_embedding)
            logger.info(f"Embedding dimension: {self.embedding_dim}")
        except Exception as e:
            logger.error("Unable to determine embedding dimension")
            raise RuntimeError("Unknown embedding dimensions") from e

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embedding for textual documents"""
        if not texts:
            return []

        sanitized_texts = [text or FALLBACK_TEXT_FOR_EMPTY_DOC for text in texts]
        
        try:
            return self.model.get_text_embedding_batch(sanitized_texts, show_progress=False)
        except Exception as e:
            logger.error(f"Batch embedding error: {e}")
            return [self.embed_query(text) for text in sanitized_texts]

    def embed_query(self, text: str) -> List[float]:
        """Embedding for single query"""
        try:
            return self.model.get_text_embedding(text or FALLBACK_TEXT_FOR_EMPTY_DOC)
        except Exception as e:
            logger.error(f"Query embedding error: {e}")
            return [0.0] * self.embedding_dim

def get_embedding_model() -> AdvancedEmbedder:
    """Factory function to create a pre-configured instance"""
    return AdvancedEmbedder()