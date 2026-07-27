from langchain_groq import ChatGroq
from src.config import (
    GROQ_API_KEY, LLM_MODEL_NAME,
    TABLE_SUMMARY_MODEL_LG, TEXT_SUMMARY_MODEL_LG, TEXT_SUMMARY_MODEL_LG
)
from pydantic import SecretStr
from typing import Optional

def get_groq_llm(model_name: Optional[str] = None):
    """
    Initialize and return Groq LLM for LangChain.
    
    Args:
        model_name: Specific model name to use. If None, uses LLM_MODEL_NAME
    """
    if not GROQ_API_KEY:
        raise ValueError("Groq API key has not been set (GROQ_API_KEY).")
    
    return ChatGroq(
        model=model_name or LLM_MODEL_NAME,
        api_key=SecretStr(GROQ_API_KEY),
        max_tokens=1000,
        stop_sequences=["<|endoftext|>"],
    )

def get_table_summary_llm():
    """Returns specific LLM for table summary (LG version)"""
    return get_groq_llm(TABLE_SUMMARY_MODEL_LG)

def get_text_summary_llm():
    """Returns specific LLM for text summary (LG version)"""
    return get_groq_llm(TEXT_SUMMARY_MODEL_LG)

def get_text_rewrite_llm():
    """Returns specific LLM for text rewriting (LG version)"""
    return get_groq_llm(TEXT_SUMMARY_MODEL_LG)
