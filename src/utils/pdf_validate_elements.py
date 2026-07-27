import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def is_valid_image(width: int, height: int) -> bool:
    print("="* 50)
    print("Validating image... INVOCATO")
    print("="* 50)
    """
    Filter for valid images based on dimensions and quality:
    - Minimum dimensions: 120x120 pixel
    - Minimum area: 14,400 pixel
    - Reasonable maximum dimensions: 5000x5000 pixel
    - Reasonable aspect ratio (not too stretched)
    """
    # Basic dimension checks
    if width < 120 or height < 120:
        logger.debug(f"Image discarded: dimensions too small ({width}x{height})")
        return False
    
    # Minimum area check
    area = width * height
    if area < 14400:  # ~120x120 pixel
        logger.debug(f"Image discarded: area too small ({area} pixels)")
        return False
    
    # Maximum dimensions check
    if width > 5000 or height > 5000:
        logger.debug(f"Image discarded: dimensions too large ({width}x{height})")
        return False
    
    # Aspect ratio check (avoid overly stretched images)
    aspect_ratio = max(width, height) / min(width, height)
    if aspect_ratio > 10:  # Maximum ratio 10:1
        logger.debug(f"Image discarded: aspect ratio too extreme ({aspect_ratio:.2f})")
        return False
    
    return True

def is_valid_table(table_data: Dict[str, Any]) -> bool:
    """
    Filter for valid tables based on structure and content:
    - At least 2 rows and 2 columns
    - Non-empty content
    """
    if not table_data:
        return False
    
    # If we have metadata about table structure
    if 'rows' in table_data and 'cols' in table_data:
        rows, cols = table_data['rows'], table_data['cols']
        if rows < 2 or cols < 2:
            logger.debug(f"Table discarded: insufficient dimensions ({rows}x{cols})")
            return False
    
    # Check for non-empty content
    content = table_data.get('text', '') or table_data.get('html', '')
    if not content or len(content.strip()) < 20:
        logger.debug("Table discarded: insufficient content")
        return False
    
    return True
