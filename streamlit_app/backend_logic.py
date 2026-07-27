import os
import logging
from typing import List
from src.pipeline.indexer_service import DocumentIndexer
from src.utils.qdrant_utils import qdrant_manager
from src.config import RAW_DATA_PATH

logger = logging.getLogger(__name__)

def process_uploaded_file(uploaded_file, indexer: DocumentIndexer) -> tuple[bool, str]:
    """
    Save and index an uploaded file.
    """
    try:
        file_name = uploaded_file.name
        save_path = os.path.join(RAW_DATA_PATH, file_name)

        # Save the file
        os.makedirs(RAW_DATA_PATH, exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Index the file
        if uploaded_file.type == "application/pdf":
            indexer.index_files([save_path], force_recreate=False)
            msg = f"File '{file_name}' indexed successfully."
            logger.info(msg)
            return True, msg
        else:
            # Here you could extend to also handle images
            raise NotImplementedError("Currently, only PDF format is supported.")

    except Exception as e:
        logger.error(f"Error processing '{uploaded_file.name}': {e}", exc_info=True)
        return False, f"Error indexing: {e}"

def delete_source(filename: str) -> tuple[bool, str]:
    """
    Manage the deletion of a source from Qdrant and the disk. Does not contain UI code.
    Returns (success, message).
    """
    try:
        # Delete from Qdrant
        success_qdrant, msg_qdrant = qdrant_manager.delete_by_source(filename)
        if not success_qdrant:
            raise Exception(f"Qdrant error: {msg_qdrant}")

        # Delete from disk
        file_path = os.path.join(RAW_DATA_PATH, filename)
        if os.path.exists(file_path):
            os.remove(file_path)

        msg = f"Source '{filename}' deleted successfully."
        logger.info(msg)
        return True, msg

    except Exception as e:
        logger.error(f"Error deleting '{filename}': {e}", exc_info=True)
        return False, f"Error deleting: {e}"

def smart_search_query(query: str, selected_files: List[str] = []) -> dict:
    """
    Utility function to test smart_query directly from backend.
    
    Args:
        query: Search query
        selected_files: Specific files to search
    
    Returns:
        Smart query results with metadata
    """
    logger.info(f"Executing smart search for: '{query}'")
    
    try:
        results = qdrant_manager.smart_query(
            query=query,
            selected_files=selected_files,
            content_types=["text", "images", "tables"]
        )
        
        # Log results for debugging
        if "query_metadata" in results:
            metadata = results["query_metadata"]
            logger.info(f"Smart search completed: intent='{metadata.get('intent')}', "
                       f"total={metadata.get('total_results')}, "
                       f"strategy='{metadata.get('search_strategy')}'")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in smart_search_query: {e}")
        return {"error": str(e)}