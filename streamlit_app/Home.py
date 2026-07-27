import sys
import os
import streamlit as st
import logging
# --- Bootstrap to resolve imports ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from streamlit_app.styles import get_custom_css

# --- Imports from new architecture ---

from streamlit_app.components.ui_components import upload_widget, source_selector_widget, enhanced_chat_interface_widget
from src.pipeline.indexer_service import DocumentIndexer
from src.utils.embedder import get_embedding_model

# --- Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(layout="wide", page_title="Scientific Papers RAG", page_icon="🔬")
st.markdown(get_custom_css(), unsafe_allow_html=True)

# --- Critical Services Loading (with caching) ---
@st.cache_resource
def load_main_services():
    """Load heavy services only once at app startup."""
    logger.info("Initializing services (embedder and indexer with semantic chunking)...")
    try:
        embedder = get_embedding_model()
        # Use global configuration for semantic chunking
        indexer = DocumentIndexer(embedder=embedder)
        logger.info(f"Services initialized successfully.")
        return indexer
    except Exception as e:
        logger.error(f"Critical error during services loading: {e}", exc_info=True)
        return None, None

# --- Application Startup ---
indexer = load_main_services()

if not indexer:
    st.error("Critical error: unable to load base services. The application cannot continue. Check logs for details.")
    st.stop()

st.title("Scientific Papers Research Assistant")


with st.sidebar:
    upload_widget(indexer)
    st.markdown("<hr>", unsafe_allow_html=True)
    selected_sources = source_selector_widget()


enhanced_chat_interface_widget(selected_sources=selected_sources)
