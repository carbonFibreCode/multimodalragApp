import logging
from typing import Dict, Any, Optional, List
from langchain.schema.messages import HumanMessage
from src.config import GROQ_API_KEY, TABLE_SUMMARY_MODEL_LG
from src.llm.groq_client import get_table_summary_llm

logger = logging.getLogger(__name__)

def create_table_summary(table_html: str, context_info: Optional[Dict[str, str]] = None, table_id: Optional[str] = None) -> str:
    """
    Generates an intelligent summary of a table using LLM.
    
    Args:
        table_html: HTML representation of the table
        context_info: Context information (caption, context_text)
        
    Returns:
        Textual summary of the table
    """
    try:
        if not GROQ_API_KEY:
            logger.warning("GROQ_API_KEY not configured, table summary not available")
            return "Summary not available due to missing configuration."

        # Build prompt for summary
        table_identifier = f"[{table_id}] " if table_id else ""
        prompt_parts = [
            f"Analyze the following table {table_identifier} and provide a concise and informative summary that includes:",
            "- The type of data contained",
            "- The main trends or patterns", 
            "- Key or interesting values",
            "- The context or purpose of the table (if evident)",
            "",
            f"TABLE {table_identifier}, here is the table in HTML format: {table_html}"
        ]
        
        # Add context information if available
        if context_info:
            if context_info.get('summary'):
                prompt_parts.extend([
                    "",
                    f"EXISTING SUMMARY: {context_info['summary']}"
                ])
        
        prompt_parts.extend([
            "",
            f"Provide a clear and structured summary in English for {table_identifier}:",
        ])
        
        prompt = "\n".join(prompt_parts)
        
        # Use specific model for table summary
        llm = get_table_summary_llm()
        
        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            summary = str(response.content).strip() if response.content else ""
            
            if summary and len(summary) > 10:
                logger.info(f"Table summary generated successfully using {TABLE_SUMMARY_MODEL_LG}")
                return summary
            else:
                logger.warning("Table summary empty or too short")
            
            return "Summary not available due to insufficient content."
        
        except Exception as llm_error:
            logger.error(f"Error during table summary generation with LLM: {llm_error}")
            
            return "Summary not available due to LLM error."
    
    except Exception as e:
        logger.error(f"Error in table summary: {e}")
        
        return "Summary not available due to internal error."
    

def enhance_table_with_summary(table_element: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enriches a table element with AI summary.
    
    Args:
        table_element: Table element with table_html and metadata

    Returns:
        Table element enriched with summary in metadata
    """
    try:
        table_html = table_element.get('table_html', '')
        metadata = table_element.get('metadata', {})
        
        # Extract table identifier
        table_id = metadata.get('table_id')
        
        # Extract context information from existing metadata
        context_info = {
            'summary': metadata.get('table_summary')
        }
        
        # Generate summary with identifier
        summary = create_table_summary(table_html, context_info, table_id)
        
        # Add summary to metadata
        metadata['table_summary'] = summary
        table_element['metadata'] = metadata
        
        logger.info(f"Table enriched with summary: {summary[:100]}...")
        return table_element
        
    except Exception as e:
        logger.error(f"Error enriching table with summary: {e}")
        # Return original element in case of error
        return table_element