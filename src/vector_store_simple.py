"""
Simplified vector store implementation using pandas and numpy.
Alternative implementation that doesn't rely on Pathway.
"""

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Tuple
import pickle
import os


class SimpleVectorStore:
    """Simplified vector store for semantic indexing using pandas and numpy."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the vector store.
        
        Args:
            model_name: Name of the sentence transformer model
        """
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.chunks_df = None
        self.embeddings = None
    
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts to embeddings.
        
        Args:
            texts: List of text strings to encode
            
        Returns:
            numpy array of embeddings
        """
        return self.model.encode(texts, show_progress_bar=True)
    
    def create_embeddings_dataframe(self, chunks: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Create embeddings dataframe from chunks.
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            DataFrame with chunks and embeddings
        """
        # Extract texts
        texts = [chunk["text"] for chunk in chunks]
        
        # Create embeddings
        print("Creating embeddings...")
        embeddings = self.encode_texts(texts)
        
        # Create DataFrame
        df = pd.DataFrame(chunks)
        df["embedding"] = [emb.tolist() for emb in embeddings]
        
        self.chunks_df = df
        self.embeddings = embeddings
        
        return df
    
    def semantic_search(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform semantic search for relevant chunks.
        
        Args:
            query_text: Query text (backstory)
            top_k: Number of top results to return
            
        Returns:
            List of relevant chunks with similarity scores
        """
        if self.chunks_df is None:
            raise ValueError("No chunks loaded. Call create_embeddings_dataframe() first.")
        
        # Encode query
        query_embedding = self.model.encode(query_text)
        
        # Compute cosine similarities
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            chunk_data = self.chunks_df.iloc[idx].to_dict()
            results.append({
                "chunk_id": chunk_data["chunk_id"],
                "novel_name": chunk_data["novel_name"],
                "text": chunk_data["text"],
                "similarity_score": float(similarities[idx]),
                "token_count": chunk_data["token_count"]
            })
        
        return results
    
    def batch_semantic_search(self, query_texts: List[str], top_k: int = 5) -> List[List[Dict[str, Any]]]:
        """
        Perform batch semantic search for multiple queries.
        
        Args:
            query_texts: List of query texts
            top_k: Number of top results per query
            
        Returns:
            List of results for each query
        """
        if self.chunks_df is None:
            raise ValueError("No chunks loaded. Call create_embeddings_dataframe() first.")
        
        # Encode all queries at once
        query_embeddings = self.model.encode(query_texts, show_progress_bar=True)
        
        all_results = []
        
        for query_embedding in query_embeddings:
            # Compute cosine similarities
            similarities = np.dot(self.embeddings, query_embedding) / (
                np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
            )
            
            # Get top-k indices
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            query_results = []
            for idx in top_indices:
                chunk_data = self.chunks_df.iloc[idx].to_dict()
                query_results.append({
                    "chunk_id": chunk_data["chunk_id"],
                    "novel_name": chunk_data["novel_name"],
                    "text": chunk_data["text"],
                    "similarity_score": float(similarities[idx]),
                    "token_count": chunk_data["token_count"]
                })
            
            all_results.append(query_results)
        
        return all_results
    
    def save_to_disk(self, filepath: str):
        """Save the vector store to disk."""
        if self.chunks_df is None:
            raise ValueError("No data to save")
        
        data = {
            "chunks_df": self.chunks_df,
            "embeddings": self.embeddings,
            "model_name": self.model_name
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load_from_disk(self, filepath: str):
        """Load the vector store from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File {filepath} not found")
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.chunks_df = data["chunks_df"]
        self.embeddings = data["embeddings"]
        self.model_name = data["model_name"]
