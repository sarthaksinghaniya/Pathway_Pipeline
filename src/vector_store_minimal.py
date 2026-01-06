"""
Minimal vector store implementation using only basic libraries.
Uses TF-IDF and cosine similarity instead of sentence transformers.
"""

import os
import re
import math
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from collections import Counter, defaultdict
import pickle


class MinimalVectorStore:
    """Minimal vector store using TF-IDF for semantic indexing."""
    
    def __init__(self):
        """Initialize the minimal vector store."""
        self.chunks_df = None
        self.tfidf_matrix = None
        self.feature_names = None
        self.idf_scores = None
    
    def preprocess_text(self, text: str) -> List[str]:
        """
        Simple text preprocessing.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        # Convert to lowercase and split on non-alphanumeric
        tokens = re.findall(r'\b[a-z]+\b', text.lower())
        
        # Remove very short tokens
        tokens = [token for token in tokens if len(token) > 2]
        
        return tokens
    
    def compute_tf_idf(self, documents: List[str]) -> np.ndarray:
        """
        Compute TF-IDF matrix for documents.
        
        Args:
            documents: List of document strings
            
        Returns:
            TF-IDF matrix
        """
        # Tokenize all documents
        tokenized_docs = [self.preprocess_text(doc) for doc in documents]
        
        # Build vocabulary
        vocab = set()
        for tokens in tokenized_docs:
            vocab.update(tokens)
        self.feature_names = sorted(list(vocab))
        
        # Compute document frequency
        df = defaultdict(int)
        for tokens in tokenized_docs:
            unique_tokens = set(tokens)
            for token in unique_tokens:
                df[token] += 1
        
        # Compute IDF scores
        n_docs = len(documents)
        self.idf_scores = {}
        for token in vocab:
            self.idf_scores[token] = math.log(n_docs / (1 + df[token]))
        
        # Compute TF-IDF matrix
        self.tfidf_matrix = np.zeros((len(documents), len(vocab)))
        
        for doc_idx, tokens in enumerate(tokenized_docs):
            # Compute term frequencies
            tf = Counter(tokens)
            
            # Compute TF-IDF for each term
            for token_idx, token in enumerate(self.feature_names):
                tf_score = tf[token] / len(tokens) if len(tokens) > 0 else 0
                self.tfidf_matrix[doc_idx, token_idx] = tf_score * self.idf_scores[token]
        
        return self.tfidf_matrix
    
    def create_embeddings_dataframe(self, chunks: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Create TF-IDF embeddings dataframe from chunks.
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            DataFrame with chunks and TF-IDF vectors
        """
        # Extract texts
        texts = [chunk["text"] for chunk in chunks]
        
        # Create TF-IDF matrix
        print("Creating TF-IDF embeddings...")
        self.compute_tf_idf(texts)
        
        # Create DataFrame
        df = pd.DataFrame(chunks)
        
        self.chunks_df = df
        return df
    
    def cosine_similarity(self, query_vector: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between query and all documents.
        
        Args:
            query_vector: Query TF-IDF vector
            
        Returns:
            Similarity scores
        """
        if self.tfidf_matrix is None:
            raise ValueError("No TF-IDF matrix computed")
        
        # Compute norms
        doc_norms = np.linalg.norm(self.tfidf_matrix, axis=1)
        query_norm = np.linalg.norm(query_vector)
        
        if query_norm == 0:
            return np.zeros(len(doc_norms))
        
        # Compute cosine similarity
        similarities = np.dot(self.tfidf_matrix, query_vector) / (doc_norms * query_norm + 1e-8)
        return similarities
    
    def query_to_vector(self, query: str) -> np.ndarray:
        """
        Convert query to TF-IDF vector.
        
        Args:
            query: Query string
            
        Returns:
            TF-IDF vector
        """
        if self.feature_names is None or self.idf_scores is None:
            raise ValueError("Vocabulary not built")
        
        # Tokenize query
        query_tokens = self.preprocess_text(query)
        
        # Compute term frequencies
        tf = Counter(query_tokens)
        
        # Create TF-IDF vector
        query_vector = np.zeros(len(self.feature_names))
        
        for token_idx, token in enumerate(self.feature_names):
            tf_score = tf[token] / len(query_tokens) if len(query_tokens) > 0 else 0
            query_vector[token_idx] = tf_score * self.idf_scores[token]
        
        return query_vector
    
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
        
        # Convert query to TF-IDF vector
        query_vector = self.query_to_vector(query_text)
        
        # Compute similarities
        similarities = self.cosine_similarity(query_vector)
        
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
        all_results = []
        
        for query_text in query_texts:
            results = self.semantic_search(query_text, top_k)
            all_results.append(results)
        
        return all_results
    
    def save_to_disk(self, filepath: str):
        """Save the vector store to disk."""
        if self.chunks_df is None:
            raise ValueError("No data to save")
        
        data = {
            "chunks_df": self.chunks_df,
            "tfidf_matrix": self.tfidf_matrix,
            "feature_names": self.feature_names,
            "idf_scores": self.idf_scores
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
        self.tfidf_matrix = data["tfidf_matrix"]
        self.feature_names = data["feature_names"]
        self.idf_scores = data["idf_scores"]
