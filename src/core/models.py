from enum import Enum
from typing import List, Optional, Dict, Tuple
from pydantic import BaseModel, Field


class ImageResult(BaseModel):
    image_base64: str
    metadata: dict
    score: float
    page_content: str

#COMMON METADATA
class ElementMetadata(BaseModel):
    """Common metadata for all elements, validated by Pydantic."""
    source: str
    page: int
    content_type: str


#TABLE
class TableMetadata(BaseModel):
    """Essential metadata for table elements"""
    content_type: str = Field(default="table", description="Content type (always 'table')")
    source: str = Field(..., description="Source file name")
    page: int = Field(..., description="Table page number")
    table_id: str = Field(..., description="Unique table identifier (e.g. table_1)")
    table_summary: Optional[str] = Field(None, description="AI-generated table summary")

class TableData(BaseModel): 
    """Model for structured table data"""
    cells: List[List[Optional[str]]]
    headers: List[str]
    shape: Tuple[int, int]

class TableElement(BaseModel):
    """Model for table elements extracted from PDFs"""
    table_html: str = Field(..., description="HTML representation of the table")
    metadata: TableMetadata = Field(..., description="Standardized table metadata")

#TEXT
class TextMetadata(ElementMetadata):
    pass  # Inherits all necessary fields from ElementMetadata

class TextElement(BaseModel):
    """Model for a text element."""
    text: str
    metadata: TextMetadata

#IMAGES
class ImageMetadata(ElementMetadata):
    content_type: str = Field(default="image", description="Content type (always 'image')")
    image_id: str = Field(..., description="Unique image identifier (e.g. image_1)")
    image_caption: Optional[str] = Field(None, description="AI-generated caption combined with context")

class ImageElement(BaseModel):
    image_base64: str  # Base64 encoded image content
    metadata: ImageMetadata


#RETRIEVER RESULT
class RetrievalResult(BaseModel):
    answer: str
    source_documents: List[Dict]
    confidence_score:float
    query_time_ms: Optional[int] = Field(None, description="Query execution time in milliseconds")
    retrieved_count: Optional[int] = Field(None, description="Number of retrieved documents")
    filters_applied: Optional[Dict] = Field(None, description="Filters applied to search")