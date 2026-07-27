from typing import List, Optional
import logging
from langchain.schema.messages import HumanMessage
import time

from src.core.models import RetrievalResult
from src.core.prompts import create_prompt_template
from src.utils.qdrant_utils import qdrant_manager
from src.llm.groq_client import get_groq_llm
from src.config import MAX_GLOBAL_DOCUMENTS

logger = logging.getLogger(__name__)


def enhanced_rag_query(query: str,
                       selected_files: Optional[List[str]] = None) -> RetrievalResult:
    """
    Execute a smart query to retrieve relevant documents and generate an answer using RAG 
    
    Args:
        query: question to ask
        selected_files: optional list of specific files to filter results
    
    Returns:
        RetrievalResult containing the answer, source documents, and metadata
    """
    logger.info(f"SMART RAG query: '{query}'")
    start_time = time.time()
    
    try:
        # Execute the smart query using Qdrant manager
        search_results = qdrant_manager.smart_query(
            query=query, # query to search
            selected_files=selected_files or [], # specific files to filter results
            content_types=["text", "images", "tables"] # types of content to retrieve
        )
        
        
        query_metadata = search_results.get("query_metadata", {})
        detected_intent = query_metadata.get("intent", "unknown")
        total_results = query_metadata.get("total_results", 0)
        specific_content_detected = query_metadata.get("specific_content_detected", None)
        content_types_used = query_metadata.get("content_types_used", ["text", "images", "tables"])
        
        logger.info(f"Query completed: intent='{detected_intent}', "
                   f"risultati={total_results} per query: '{query}'")

        # Combine results from different content types with intelligent balancing
        all_results = []
        
        # Collect results from each content type
        text_results = search_results.get("text", [])
        image_results = search_results.get("images", [])
        table_results = search_results.get("tables", [])
        
        # Calculate balanced limits based on available content types
        total_types_with_content = sum([
            1 if text_results else 0,
            1 if image_results else 0, 
            1 if table_results else 0
        ])
        
        if total_types_with_content > 0:
            # Distribute MAX_GLOBAL_DOCUMENTS among content types
            base_limit_per_type = max(1, MAX_GLOBAL_DOCUMENTS // total_types_with_content)
            remaining_slots = MAX_GLOBAL_DOCUMENTS - (base_limit_per_type * total_types_with_content)
            
            logger.info(f"Balancing results: {base_limit_per_type} per type, "
                       f"{remaining_slots} extra slots, total types: {total_types_with_content}")
        else:
            base_limit_per_type = MAX_GLOBAL_DOCUMENTS
            remaining_slots = 0
        
        
        # Add results from text search (with limit)
        text_limit = base_limit_per_type + (1 if remaining_slots > 0 and text_results else 0)
        for text_result in text_results[:text_limit]:
            all_results.append({
                "content": text_result["content"],
                "metadata": text_result["metadata"],
                "score": text_result["score"],
                "content_type": "text",
                "relevance_tier": text_result.get("relevance_tier", "medium")
            })
        if remaining_slots > 0 and text_results:
            remaining_slots -= 1
        
        # Add results from image search (with limit)
        image_limit = base_limit_per_type + (1 if remaining_slots > 0 and image_results else 0)
        for img_result in image_results[:image_limit]:
            all_results.append({
                "content": img_result.page_content,
                "metadata": img_result.metadata,
                "score": img_result.score,
                "content_type": "image",
                "relevance_tier": "medium",  # Default value since ImageResult doesn't have this field
                "image_base64": img_result.image_base64  # Add the base64 data
            })
        if remaining_slots > 0 and image_results:
            remaining_slots -= 1
        
        # Add results from table search (with limit)
        table_limit = base_limit_per_type + (1 if remaining_slots > 0 and table_results else 0)
        for table_result in table_results[:table_limit]:
            all_results.append({
                "content": table_result["page_content"],
                "metadata": table_result["metadata"], 
                "score": table_result["score"],
                "content_type": "table",
                "relevance_tier": table_result.get("relevance_tier", "medium")
            })
        
        
        logger.info(f"Results after balancing: {len(all_results)} total chunks "
                   f"(text: {len([r for r in all_results if r['content_type'] == 'text'])}, "
                   f"images: {len([r for r in all_results if r['content_type'] == 'image'])}, "
                   f"tables: {len([r for r in all_results if r['content_type'] == 'table'])})")

        # Build the final documents list with metadata and content
        documents = []
        for result in all_results:
            doc_info = {
                "metadata": result["metadata"],
                "source": result["metadata"].get("source", "Sconosciuto"),
                "page": result["metadata"].get("page", "N/A"),
                "content_type": result["content_type"],
                "score": result["score"],
                "relevance_tier": result.get("relevance_tier", "medium")
            }
            
            if result["content_type"] == "table":
                doc_info.update({
                    "content": result["content"],
                    "caption": result["metadata"].get("caption"),
                    "context_text": result["metadata"].get("context_text", ""),
                    "table_html_raw": result["metadata"].get("table_html")
                })
            elif result["content_type"] == "image":
                doc_info.update({
                    "content": result["content"],
                    "image_base64": result.get("image_base64"), # Base64 encoded image content to be used in LLM
                    "image_caption": result["metadata"].get("image_caption")
                })
            else:
                # Assuming text content is a string
                content = result["content"]
                # Limit content length for display
                doc_info["content"] = content[:500] + "..." if len(content) > 500 else content
            
            # Add information about the document
            documents.append(doc_info)
        
        # Sort documents by score in descending order
        documents.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        # Final safety check - should not be needed but just in case
        if len(documents) > MAX_GLOBAL_DOCUMENTS:
            logger.warning(f"Safety limit triggered: reducing documents from {len(documents)} to {MAX_GLOBAL_DOCUMENTS}")
            documents = documents[:MAX_GLOBAL_DOCUMENTS]
        
        # Log the content types and detected intent  
        content_types = {}
        for doc in documents:
            doc_type = doc.get("content_type", "unknown")
            content_types[doc_type] = content_types.get(doc_type, 0) + 1
        
        logger.info(f"Content retrieved by type: {content_types}")
        logger.info(f"Intent: '{detected_intent}' - Strategy: {query_metadata.get('search_strategy', 'N/A')}")
        logger.info(f"FINAL: Using {len(documents)} documents out of max {MAX_GLOBAL_DOCUMENTS} allowed")
        
        # Documents to LLM to generate response
        # Unified context for LLM
        context_texts = []
        for doc in documents[:len(documents)]:
            # Extract relevant metadata and content
            doc_type = doc.get("content_type", "text")
            source = doc.get("source", "Unknown")
            page = doc.get("page", "N/A")
            content = doc.get("content", "")
            relevance_tier = doc.get("relevance_tier", "medium")
            
            if doc_type == "table":
                table_id = doc.get("metadata", {}).get("table_id", "")
                identifier = f"[{table_id}] " if table_id else ""
                context_texts.append(f"[TABELLA {identifier}da {source}, pagina {page}] (Rilevanza: {relevance_tier})\n{content}\n")
            elif doc_type == "image":
                image_id = doc.get("metadata", {}).get("image_id", "")
                image_caption = doc.get("metadata", {}).get("image_caption", "")
                context_text = doc.get("metadata", {}).get("context_text", "")
                image_description = doc.get("metadata", {}).get("image_description", "")
                
                identifier = f"[{image_id}] " if image_id else ""
                
                # Build comprehensive image context
                image_info_parts = []
                if image_caption:
                    image_info_parts.append(f"Caption: {image_caption}")
                if image_description:
                    image_info_parts.append(f"Description: {image_description}")
                if context_text:
                    image_info_parts.append(f"Context: {context_text}")
                if content:
                    image_info_parts.append(f"Content: {content}")
                
                image_info = " | ".join(image_info_parts) if image_info_parts else "No description available"
                
                context_texts.append(f"[IMMAGINE {identifier}da {source}, pagina {page}]\n{image_info}\n")
            else:
                context_texts.append(f"[TESTO da {source}, pagina {page}]\n{content}\n")
        
        unified_context = "\n".join(context_texts)
        
        # GENERATION OF THE ANSWER
        llm = get_groq_llm()
        # PROMPT TEMPLATE 
        prompt = create_prompt_template()
        
        # Add specific instructions for content-specific queries
        if specific_content_detected == "images":
            specific_instruction = "\n\nIMPORTANTE: L'utente ha chiesto specificamente informazioni sulle IMMAGINI. Concentrati solo sulle immagini trovate e ignora tabelle o testo. Fornisci un elenco chiaro delle immagini con le loro descrizioni e caratteristiche."
        elif specific_content_detected == "tables":
            specific_instruction = "\n\nIMPORTANTE: L'utente ha chiesto specificamente informazioni sulle TABELLE. Concentrati solo sulle tabelle trovate e ignora immagini o testo normale."
        else:
            specific_instruction = ""
        
        #Creating prompt with langchain format
        formatted_prompt = prompt.format(context=unified_context, input=query + specific_instruction)
        # Invoke the LLM with the formatted prompt
        response = llm.invoke([HumanMessage(content=formatted_prompt)])
        
        # Extract the answer from the response
        if hasattr(response, 'content'):
            answer = str(response.content)
        else:
            answer = str(response)
        
        # METRICS 
        confidence_score = min(1.0, len(documents) / 10.0)  # Based on retrieved documents
        
        query_time_ms = int((time.time() - start_time) * 1000)
        
        return RetrievalResult(
            # Final result with answer and metadata
            answer=answer,
            source_documents=documents,
            confidence_score=confidence_score,
            query_time_ms=query_time_ms,
            retrieved_count=len(documents),
            filters_applied={
                "selected_files": selected_files, 
                "smart_query": True,
                "detected_intent": detected_intent,
                "content_types_found": content_types,
                "content_types_used": content_types_used,
                "specific_content_detected": specific_content_detected,
                "search_strategy": query_metadata.get("search_strategy", "N/A")
            }
        )

    except Exception as e:
        logger.error(f"Errore RAG: {e}")
        query_time_ms = int((time.time() - start_time) * 1000)
        return RetrievalResult(
            answer="Error during search",
            source_documents=[],
            confidence_score=0.0,
            query_time_ms=query_time_ms,
            retrieved_count=0,
            filters_applied={"error": str(e)}
        )