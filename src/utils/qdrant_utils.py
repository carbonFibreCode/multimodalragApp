from typing import List, Optional, Dict, Any, Tuple, Union
import logging
import qdrant_client
from qdrant_client.http import models
from src.config import (
    QDRANT_URL, COLLECTION_NAME,
    SCORE_THRESHOLD_TEXT, SCORE_THRESHOLD_IMAGES, SCORE_THRESHOLD_TABLES, 
    RAG_PARAMS, ADAPTIVE_K_MIN, ADAPTIVE_K_MAX
)
from src.core.models import ImageResult, TextElement, ImageElement, TableElement
from src.utils.embedder import get_embedding_model
import uuid

logger = logging.getLogger(__name__)

class QdrantManager:
    """
    Manages all operations with the Qdrant vector database.
    Centralizes connection, collection creation, insertion and search.
    """
    
    def __init__(self, 
                 url: str = QDRANT_URL, 
                 collection_name: str = COLLECTION_NAME):
        self.url = url
        self.collection_name = collection_name
        self._client = None
        self._embedder = None
    
    @property
    def client(self) -> qdrant_client.QdrantClient:
        if self._client is None:
            self._client = qdrant_client.QdrantClient(
                url=self.url,
                prefer_grpc=True,
                timeout=60
            )
        return self._client
    
    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = get_embedding_model()
        return self._embedder
    
    # === MODELS TO POINT IN QDRANT ===
    
    def _text_element_to_point(self, 
                               element: TextElement, 
                               vector: List[float]) -> models.PointStruct:
        return models.PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={
                "page_content": element.text,
                "content_type": "text",
                "metadata": element.metadata.model_dump(),
            }
        )
    
    def _image_element_to_point(self,
                                element: Union[ImageElement, Dict[str, Any]], 
                                vector: List[float]) -> models.PointStruct:
    
        if isinstance(element, dict):
            metadata = element.get("metadata", {})
            return models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "page_content": element.get("page_content", ""),
                    "content_type": "image",
                    "image_base64": element.get("image_base64", ""),
                    "metadata": metadata
                }
            )
        else:
            #If pydantic object
            return models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "page": element.metadata.page,
                    "content_type": "image",
                    "image_base64": element.image_base64,
                    "metadata": element.metadata.model_dump()
                }
            )
    
    def _table_element_to_point(self, 
                                element: Union[TableElement, Dict[str, Any]], 
                                vector: List[float]) -> models.PointStruct:
        
        if isinstance(element, dict):
            return models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "page_content": element.get("table_html", ""),
                    "metadata": element.get("metadata", {}),
                    "content_type": "table",
                }
            )
        else:
            return models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "page_content": element.table_html,
                    "metadata": element.metadata.model_dump(),
                    "content_type": "table",
                }
            )
    
    def convert_elements_to_points(
        self,
        elements: List[Any],
        vectors: List[List[float]]
    ) -> List[models.PointStruct]:
        """
        Converts an elements list and vectors list into Qdrant points for insertion.
        """
        points = []
        for element, vector in zip(elements, vectors):
            if isinstance(element, TextElement):
                points.append(self._text_element_to_point(element, vector))
            elif isinstance(element, (ImageElement, dict)) and (
                hasattr(element, 'image_base64') or 
                (isinstance(element, dict) and 'image_base64' in element)
            ):
                points.append(self._image_element_to_point(element, vector))
            elif isinstance(element, (TableElement, dict)) and (
                hasattr(element, 'table_html') or 
                (isinstance(element, dict) and 'table_html' in element)
            ):
                points.append(self._table_element_to_point(element, vector))
            else:
                logger.warning(f"Non recognizible element fo the insertion: {element}")
        return points
    
    # === COLLECTION AND CONNESSION HANDLING ===
    
    def verify_connection(self) -> bool:
        try:
            self.client.get_collections()
            logger.info(f"Qdrant connected at {self.url}")
            return True
        except Exception as e:
            logger.error(f"Failed connection: {e}")
            return False
    
    def collection_exists(self) -> bool:
        try:
            return self.client.collection_exists(self.collection_name)
        except Exception as e:
            logger.error(f"Collection verification error: {e}")
            return False

    def create_collection(self, 
                          embedding_dim: int, 
                          force_recreate: bool = False) -> bool:
        try:
            if force_recreate and self.collection_exists():
                self.delete_collection()
                logger.info(f"Collection {self.collection_name} deleted for recreation")
            if not self.collection_exists():
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=embedding_dim,
                        distance=models.Distance.COSINE,
                        on_disk=True
                    )
                )
                logger.info(f"Collection {self.collection_name} created successfully")
                return True
            else:
                logger.info(f"Collection {self.collection_name} already exists")
                return True
        except Exception as e:
            logger.error(f"Collection creation error: {e}")
            return False
    
    def delete_collection(self) -> bool:
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Collection {self.collection_name} deleted")
            return True
        except Exception as e:
            logger.error(f"Collection deletion error: {e}")
            return False
    
    def ensure_collection_exists(self, 
                                 embedding_dim: int) -> bool:
        if not self.collection_exists():
            return self.create_collection(embedding_dim)
        return True
    
    # === CRUD OPERATIONS ===
    
    def upsert_points(self, 
                      points: List[models.PointStruct], 
                      batch_size: int = 64) -> bool:
        try:
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch,
                    wait=True
                )
            logger.info(f"Inserted {len(points)} points")
            return True
        except Exception as e:
            logger.error(f"Point insertion error: {e}")
            return False
    
    def delete_by_source(self, 
                         filename: str) -> Tuple[bool, str]:
        logger.info(f"Deleting documents for source='{filename}'")
        try:
            # Create filter to match both metadata.source and source
            qdrant_filter = models.Filter(
                should=[
                    models.FieldCondition(
                        key="metadata.source",
                        match=models.MatchValue(value=filename),
                    ),
                    models.FieldCondition(
                        key="source",
                        match=models.MatchValue(value=filename),
                    ),
                ],
            )
            # Perform deletion
            response = self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(filter=qdrant_filter),
                wait=True
            )
            logger.info(f"Deletion result: {response}")
            return True, "Points successfully deleted from Qdrant"
        except Exception as e:
            error_message = f"Error during deletion from Qdrant: {e}"
            logger.error(error_message)
            return False, error_message
    
    # === FILTERS ===
    
    def create_content_filter(self, 
                              query_type: Optional[str] = None) -> Optional[models.Filter]:
        # Create a filter based on content type
        if query_type == "image":
            return models.Filter(
                must=[
                    models.FieldCondition(
                        key="content_type",
                        match=models.MatchValue(value="image"),
                    )
                ]
            )
        elif query_type == "text":
            return models.Filter(
                must=[
                    models.FieldCondition(
                        key="content_type",
                        match=models.MatchValue(value="text"),
                    )
                ]
            )
        elif query_type == "table":
            return models.Filter(
                must=[
                    models.FieldCondition(
                        key="content_type",
                        match=models.MatchValue(value="table"),
                    )
                ]
            )
        return None
    
    def create_file_filter(self, 
                           selected_files: List[str]) -> Optional[models.Filter]:
        """ Filter creation for selected files"""
        if not selected_files:
            return None
        
        # Create conditions for each selected file
        file_conditions = []
        for filename in selected_files:
            file_conditions.extend([
                models.FieldCondition(
                    key="metadata.source",
                    match=models.MatchValue(value=filename),
                ),
                models.FieldCondition(
                    key="source",
                    match=models.MatchValue(value=filename),
                )
            ])
        
        return models.Filter(should=file_conditions)

    def build_combined_filter(self, 
                            selected_files: List[str] = [],
                            query_type: Optional[str] = None) -> Optional[models.Filter]:
        """Combines file and content filters into a single Qdrant filter."""
        filters = []
        # Create file filter if files are selected
        if selected_files:
            file_filter = self.create_file_filter(selected_files)
            if file_filter:
                filters.append(file_filter)

        # Create content filter if query type is specified
        if query_type:
            content_filter = self.create_content_filter(query_type)
            if content_filter:
                filters.append(content_filter)

        # If no filters are specified, return None
        if not filters:
            return None
        if len(filters) == 1:
            return filters[0]
        return models.Filter(must=filters)
    
    # === OPTIMIZED SEARCH ===
    
    def get_optimal_search_params(self, 
                                  query_type: str = "multimodal",
                                  query_intent: str = "exploratory") -> Dict[str, Any]:
        """
        Returns optimized parameters based on query type and intent.
        """
        base_params = RAG_PARAMS.get(query_intent, RAG_PARAMS["multimodal"])
        
        #Different score thresold for eeach query type
        if query_type == "text":
            score_threshold = SCORE_THRESHOLD_TEXT
        elif query_type == "image":
            score_threshold = SCORE_THRESHOLD_IMAGES
        elif query_type == "table":
            score_threshold = SCORE_THRESHOLD_TABLES
        else:
            score_threshold = base_params["score_threshold"]
        
        # Return optimized parameters 
        return {
            "k": base_params["k"],
            "score_threshold": score_threshold,
            "description": base_params["description"]
        }
    
    def search_vectors_adaptive(self, 
                               query_embedding: List[float],
                               query_type: Optional[str] = None,
                               query_intent: str = "exploratory",
                               selected_files: List[str] = [],
                               custom_k: Optional[int] = None,
                               custom_threshold: Optional[float] = None) -> List[models.ScoredPoint]:
        """
        Vector search with adaptive parameters based on query type and intent.
        """
        try:
            # Get optimized parameters from previous function
            optimal_params = self.get_optimal_search_params(query_type or "multimodal", query_intent)
            
            # Use custom parameters if provided, otherwise use optimal ones
            k = custom_k or optimal_params["k"]
            score_threshold = custom_threshold or optimal_params["score_threshold"]
            
            # Ensure k is within the defined limits
            k = max(ADAPTIVE_K_MIN, min(k, ADAPTIVE_K_MAX))
            
            qdrant_filter = self.build_combined_filter(selected_files, query_type)

            logger.info(f"Adaptive search: k={k}, threshold={score_threshold:.2f}, "
                       f"intent='{query_intent}', type='{query_type}'")
            
            # Perform the search with the adaptive parameters
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                query_filter=qdrant_filter,
                limit=k,
                with_payload=True,
                with_vectors=False,
                score_threshold=score_threshold
            )
            
            logger.info(f"Adaptive search: found {len(results)} results "
                       f"(threshold: {score_threshold:.2f}, {optimal_params['description']})")
            return results
            
        except Exception as e:
            logger.error(f"Adaptive vector search error: {e}")
            return []
    
    def query_text(self, 
                   query: str, 
                   selected_files: List[str] = [],
                   query_intent: str = "exploratory",
                   top_k: Optional[int] = None,
                   score_threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Optimized text query with adaptive parameters.
        
        Args:
            query: Query text
            selected_files: File names to filter results
            query_intent: Intent of the query (e.g., "factual", "exploratory", "technical", "multimodal")
            top_k: Number of results (None to use optimal)
            score_threshold: similarity thresold (None to use optimal)
        """
        logger.info(f"Optimzied query text: '{query}' (intent: {query_intent})")
           
        try:
            # Embed the query using the embedder
            query_embedding = self.embedder.embed_query(query)
            # Perform the search with adaptive parameters
            results = self.search_vectors_adaptive(
                query_embedding=query_embedding,
                query_type="text",
                query_intent=query_intent,
                selected_files=selected_files,
                custom_k=top_k,
                custom_threshold=score_threshold
            )
            
            text_results = []
            for result in results:
                try:
                    payload = result.payload or {}
                    metadata = payload.get("metadata", {})
                    content = payload.get("page_content", "")
                    
                    if not content:
                        logger.debug(f"Saltato risultato senza contenuto testuale: {result.id}")
                        continue
                    
                    text_results.append({
                        "content": content,
                        "metadata": metadata,
                        "score": result.score,
                        "source": metadata.get("source", "Unknown"),
                        "page": metadata.get("page", "N/A"),
                        "content_type": payload.get("content_type", "text"),
                        "relevance_tier": "high" if result.score > 0.80 else "medium" if result.score > 0.65 else "low"
                    })
                except Exception as e:
                    logger.warning(f"Errore processamento risultato testo: {e}")
                    continue
            
            logger.info(f"Trovati {len(text_results)} documenti di testo per query '{query}' "
                       f"(intent: {query_intent})")
            return text_results
        except Exception as e:
            logger.error(f"Errore query testo: {e}")
            return []
    
    def query_images(self, 
                    query: str, 
                    selected_files: List[str] = [],
                    query_intent: str = "exploratory",
                    top_k: Optional[int] = None) -> List[ImageResult]:
        logger.info(f"Optimized query: '{query}' (intent: {query_intent})")
        try:
            # Embed the query using the embedder
            query_embedding = self.embedder.embed_query(query)
            # Perform the search with adaptive parameters
            results = self.search_vectors_adaptive(
                query_embedding=query_embedding,
                query_type="image",
                query_intent=query_intent,
                selected_files=selected_files,
                custom_k=top_k
            )
            
            image_results = []
            for result in results:
                try:
                    # Extract payload and metadata
                    payload = result.payload or {}
                    metadata = payload.get("metadata", {})
                    image_base64 = payload.get("image_base64", "")
                    
                    if not image_base64:
                        logger.debug(f"Jumped result w/out base64: {result.id}")
                        continue
                    
                    image_results.append(ImageResult( 
                        image_base64=image_base64,
                        metadata=metadata,
                        score=result.score,
                        page_content=payload.get("page_content", "")
                    ))
                except Exception as e:
                    logger.warning(f"Errorprocessing image: {e}")
                    continue
            
            logger.info(f"Found {len(image_results)} images for query '{query}' (intent: {query_intent})")
            return image_results
        except Exception as e:
            logger.error(f"Error image query: {e}")
            return []
    
    def query_tables(self, 
                     query: str, 
                     selected_files: List[str] = [],
                     query_intent: str = "exploratory",
                     top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        logger.info(f"Optimized table query: '{query}' (intent: {query_intent})")
        try:
            query_embedding = self.embedder.embed_query(query)
            results = self.search_vectors_adaptive(
                query_embedding=query_embedding,
                query_type="table",
                query_intent=query_intent,
                selected_files=selected_files,
                custom_k=top_k
            )
            
            table_results = []
            for result in results:
                try:
                    payload = result.payload or {}
                    metadata = payload.get("metadata", {})
                    table_html = payload.get("page_content", "")
                    
                    if not table_html:
                        logger.debug(f"Skipped result without table content: {result.id}")
                        continue
                    
                    table_results.append({
                        "table_html": table_html,
                        "metadata": metadata,
                        "score": result.score,
                        "page_content": table_html,
                        "relevance_tier": "high" if result.score > 0.75 else "medium" if result.score > 0.60 else "low"
                    })
                except Exception as e:
                    logger.warning(f"Error processing table result: {e}")
                    continue
            
            logger.info(f"Found {len(table_results)} tables for query '{query}' (intent: {query_intent})")
            return table_results
        except Exception as e:
            logger.error(f"Table query error: {e}")
            return []

    def debug_collection_content(self, 
                                 limit: int = 10) -> Dict[str, Any]:
        """
        Debug method to view collection content.
        """
        try:
            results, _ = self.client.scroll(
                collection_name=self.collection_name,
                limit=limit,
                with_payload=True,
                with_vectors=False
            )
            
            debug_info = {
                "total_points_sampled": len(results),
                "content_types": {},
                "sources": set(),
                "sample_points": []
            }
            
            for result in results:
                payload = result.payload or {}
                content_type = payload.get("content_type", "unknown")
                
                # Count content types
                if content_type not in debug_info["content_types"]:
                    debug_info["content_types"][content_type] = 0
                debug_info["content_types"][content_type] += 1
                
                # Collect sources
                metadata = payload.get("metadata", {})
                source = metadata.get("source", payload.get("source", "unknown"))
                debug_info["sources"].add(source)
                
                # Sample points
                if len(debug_info["sample_points"]) < 5:
                    debug_info["sample_points"].append({
                        "id": result.id,
                        "content_type": content_type,
                        "source": source,
                        "page": metadata.get("page", payload.get("page", "N/A")),
                        "content_preview": payload.get("page_content", "")[:100] + "..." if payload.get("page_content") else "No content"
                    })
            
            debug_info["sources"] = list(debug_info["sources"])
            logger.info(f"Collection debug: {debug_info['content_types']}")
            return debug_info
            
        except Exception as e:
            logger.error(f"Collection debug error: {e}")
            return {"error": str(e)}
    
    def get_collection_info(self) -> Dict[str, Any]:
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "vectors_count": collection_info.vectors_count,
                "points_count": collection_info.points_count,
                "status": collection_info.status,
                "config": collection_info.config
            }
        except Exception as e:
            logger.error(f"Errore recupero info collezione: {e}")
            return {}
  
    def health_check(self) -> Dict[str, Any]:
        try:
            connection_ok = self.verify_connection()
            collection_exists = self.collection_exists()
            collection_info = self.get_collection_info() if collection_exists else {}
            debug_info = self.debug_collection_content(5) if collection_exists else {}
            
            return {
                "connection": connection_ok,
                "collection_exists": collection_exists,
                "collection_info": collection_info,
                "debug_info": debug_info,
                "url": self.url,
                "collection_name": self.collection_name
            }
        except Exception as e:
            logger.error(f"Errore health check: {e}")
            return {"error": str(e)}
    
    # === INTELLIGENT QUERY ===
    
    def detect_query_intent(self, 
                            query: str) -> str:
        """
        Automatically determines query intent by analyzing the text.
        
        Returns:
            Query intent: "factual", "exploratory", "technical", or "multimodal"
        """
        query_lower = query.lower()
        
        # Keywords for factual intent (specific questions)
        factual_keywords = [
            "cosa è", "cos'è", "che cos'è", "definisci", "definizione",
            "quando", "dove", "chi", "quale", "quanto", "quanti",
            "data", "numero", "valore", "risultato", "statistica",
            "informazione", "dettaglio", "specifica", "esempio", "spiegazione",
            "descrivi", "caratteristica", "funzione", "utilizzo", "scopo", "obiettivo",
            "what is", "what's", "define", "definition", "when", "where", "who", "which", "how many", "data", "number", "value", "result",
            "statistic", "information", "detail", "specific", "example", "explanation",
            "describe", "feature", "function", "usage", "purpose", "goal"
        ]
        
        # Keywords for technical intent (technical content)
        technical_keywords = [
            "algoritmo", "codice", "implementazione", "funzione", "metodo",
            "classe", "api", "configurazione", "parametri", "variabili",
            "sistema", "architettura", "design pattern", "framework", "libreria",
            "tecnologia", "protocollo", "rete", "database", "query",
            "performance", "ottimizzazione", "debug", "errore", "bug",
            "algorithm", "code", "implementation", "function", "method",
            "class", "api", "configuration", "parameters", "variables",
            "system", "architecture", "design pattern", "framework", "library",
            "technology", "protocol", "network", "database", "query",
            "performance", "optimization", "debug", "error", "bug"
        ]
        
        # Keywords for multimodal intent (mixed content)
        multimodal_keywords = [
            "immagine", "tabella", "grafico", "figura", "diagramma",
            "chart", "visualizzazione", "schema", "esempio visivo", "dati visivi",
            "image", "table", "graph", "figure", "diagram",
            "chart", "visualization", "schema", "visual example", "visual data",
            "multimodal", "mixed content", "text and images", "text and tables",
            "text and graphs", "text and figures", "text and diagrams"
        ]
        
        # Count occurrences for each category
        factual_score = sum(1 for keyword in factual_keywords if keyword in query_lower)
        technical_score = sum(1 for keyword in technical_keywords if keyword in query_lower)
        multimodal_score = sum(1 for keyword in multimodal_keywords if keyword in query_lower)
        
        # Determine intent based on scores
        if multimodal_score > 0:
            return "multimodal"
        elif technical_score > factual_score:
            return "technical"
        elif factual_score > 0:
            return "factual"
        else:
            return "exploratory"  # Default for generic queries
    
    def smart_query(self, 
                   query: str, 
                   selected_files: List[str] = [],
                   content_types: List[str] = ["text", "images", "tables"]) -> Dict[str, Any]:
        """
        Executes intelligent query with automatic intent detection.
        
        Args:
            query: Search query
            selected_files: Specific files to search
            content_types: Content types to include
        """
        intent = self.detect_query_intent(query)
        logger.info(f"Smart query: '{query}' -> detected intent: '{intent}'")
        
        results = {}
        
        try:
            if "text" in content_types:
                results["text"] = self.query_text(
                    query=query,
                    selected_files=selected_files,
                    query_intent=intent
                )
            
            if "images" in content_types:
                results["images"] = self.query_images(
                    query=query,
                    selected_files=selected_files,
                    query_intent=intent
                )
            
            if "tables" in content_types:
                results["tables"] = self.query_tables(
                    query=query,
                    selected_files=selected_files,
                    query_intent=intent
                )
            
            # Add query metadata
            results["query_metadata"] = {
                "intent": intent,
                "query": query,
                "total_results": sum(len(results.get(t, [])) for t in content_types),
                "search_strategy": RAG_PARAMS[intent]["description"]
            }
            
        except Exception as e:
            logger.error(f"Error in smart_query: {e}")
            results["error"] = str(e)
        
        return results

# Singleton for global use 
qdrant_manager = QdrantManager()