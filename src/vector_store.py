"""
Vector store implementation using Pathway for semantic indexing and retrieval.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
import pathway as pw
from typing import List, Dict, Any, Tuple
import pandas as pd


class SemanticVectorStore:
    """Vector store for semantic indexing using Pathway and sentence-transformers."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the vector store.
        
        Args:
            model_name: Name of the sentence transformer model
        """
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.chunks_table = None
        self.embeddings_table = None
    
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts to embeddings.
        
        Args:
            texts: List of text strings to encode
            
        Returns:
            numpy array of embeddings
        """
        return self.model.encode(texts, show_progress_bar=True)
    
    def create_embeddings_table(self, chunks_table: pw.Table) -> pw.Table:
        """
        Create embeddings table from chunks table.
        
        Args:
            chunks_table: Pathway table with chunk data
            
        Returns:
            Pathway table with embeddings
        """
        @pw.udf
        def embed_text(text: str) -> pw.Json:
            """UDF to embed text using sentence transformer."""
            embedding = self.model.encode(text)
            return pw.Json(embedding.tolist())
        
        # Apply embedding function
        embeddings_table = chunks_table.with_columns(
            embedding=embed_text(pw.this.text)
        )
        
        self.embeddings_table = embeddings_table
        return embeddings_table
    
    def setup_vector_index(self, embeddings_table: pw.Table) -> pw.Table:
        """
        Set up vector index for similarity search.
        
        Args:
            embeddings_table: Table with embeddings column
            
        Returns:
            Table with vector index capabilities
        """
        @pw.udf
        def parse_embedding(embedding_json: pw.Json) -> np.ndarray:
            """Parse JSON embedding back to numpy array."""
            return np.array(embedding_json)
        
        # Add parsed embeddings for indexing
        indexed_table = embeddings_table.with_columns(
            embedding_array=parse_embedding(pw.this.embedding)
        )
        
        return indexed_table
    
    def semantic_search(self, query_text: str, indexed_table: pw.Table, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform semantic search for relevant chunks.
        
        Args:
            query_text: Query text (backstory)
            indexed_table: Indexed table with embeddings
            top_k: Number of top results to return
            
        Returns:
            List of relevant chunks with similarity scores
        """
        # Encode query
        query_embedding = self.model.encode(query_text)
        
        # Convert table to pandas for similarity computation
        df = indexed_table.to_pandas()
        
        if df.empty:
            return []
        
        # Parse embeddings and compute similarities
        embeddings = np.array([np.array(emb) for emb in df['embedding'].values])
        
        # Compute cosine similarities
        similarities = np.dot(embeddings, query_embedding) / (
            np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            chunk_data = df.iloc[idx]
            results.append({
                "chunk_id": chunk_data["chunk_id"],
                "novel_name": chunk_data["novel_name"],
                "text": chunk_data["text"],
                "similarity_score": float(similarities[idx]),
                "token_count": chunk_data["token_count"]
            })
        
        return results
    
    def batch_semantic_search(self, query_texts: List[str], indexed_table: pw.Table, top_k: int = 5) -> List[List[Dict[str, Any]]]:
        """
        Perform batch semantic search for multiple queries.
        
        Args:
            query_texts: List of query texts
            indexed_table: Indexed table with embeddings
            top_k: Number of top results per query
            
        Returns:
            List of results for each query
        """
        # Encode all queries at once
        query_embeddings = self.model.encode(query_texts, show_progress_bar=True)
        
        # Convert table to pandas once
        df = indexed_table.to_pandas()
        
        if df.empty:
            return [[] for _ in query_texts]
        
        # Parse embeddings once
        embeddings = np.array([np.array(emb) for emb in df['embedding'].values])
        
        all_results = []
        
        for query_embedding in query_embeddings:
            # Compute cosine similarities
            similarities = np.dot(embeddings, query_embedding) / (
                np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
            )
            
            # Get top-k indices
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            query_results = []
            for idx in top_indices:
                chunk_data = df.iloc[idx]
                query_results.append({
                    "chunk_id": chunk_data["chunk_id"],
                    "novel_name": chunk_data["novel_name"],
                    "text": chunk_data["text"],
                    "similarity_score": float(similarities[idx]),
                    "token_count": chunk_data["token_count"]
                })
            
            all_results.append(query_results)
        
        return all_results
