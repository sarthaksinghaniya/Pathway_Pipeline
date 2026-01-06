"""
Simplified retrieval module using the simple vector store.
"""

from typing import List, Dict, Any, Optional
import pandas as pd
from .vector_store_simple import SimpleVectorStore


class SimpleBackstoryRetriever:
    """Handles retrieval of relevant novel chunks for given backstories."""
    
    def __init__(self, vector_store: SimpleVectorStore, top_k: int = 10, similarity_threshold: float = 0.3):
        """
        Initialize the retriever.
        
        Args:
            vector_store: Initialized simple vector store
            top_k: Number of top chunks to retrieve per backstory
            similarity_threshold: Minimum similarity threshold for chunks
        """
        self.vector_store = vector_store
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
    
    def retrieve_for_backstory(self, backstory: str, novel_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks for a single backstory.
        
        Args:
            backstory: The backstory text to query
            novel_name: Optional novel name to filter results
            
        Returns:
            List of relevant chunks with metadata
        """
        # Perform semantic search
        results = self.vector_store.semantic_search(
            query_text=backstory,
            top_k=self.top_k * 2  # Get more to filter by threshold
        )
        
        # Filter by similarity threshold
        filtered_results = [
            result for result in results 
            if result["similarity_score"] >= self.similarity_threshold
        ]
        
        # Filter by novel name if specified
        if novel_name:
            filtered_results = [
                result for result in filtered_results 
                if result["novel_name"] == novel_name
            ]
        
        # Return top-k results
        return filtered_results[:self.top_k]
    
    def retrieve_for_backstories_batch(self, backstories: List[str], novel_names: Optional[List[str]] = None) -> List[List[Dict[str, Any]]]:
        """
        Retrieve relevant chunks for multiple backstories.
        
        Args:
            backstories: List of backstory texts
            novel_names: Optional list of novel names for filtering
            
        Returns:
            List of result lists, one per backstory
        """
        # Perform batch semantic search
        all_results = self.vector_store.batch_semantic_search(
            query_texts=backstories,
            top_k=self.top_k * 2  # Get more to filter by threshold
        )
        
        # Process results for each query
        final_results = []
        for i, results in enumerate(all_results):
            # Filter by similarity threshold
            filtered_results = [
                result for result in results 
                if result["similarity_score"] >= self.similarity_threshold
            ]
            
            # Filter by novel name if specified
            if novel_names and i < len(novel_names) and novel_names[i]:
                filtered_results = [
                    result for result in filtered_results 
                    if result["novel_name"] == novel_names[i]
                ]
            
            # Return top-k results
            final_results.append(filtered_results[:self.top_k])
        
        return final_results
    
    def retrieve_from_dataframe(self, df: pd.DataFrame, content_column: str = "content", 
                               novel_column: str = "book_name") -> pd.DataFrame:
        """
        Retrieve relevant chunks for backstories in a DataFrame.
        
        Args:
            df: DataFrame containing backstories
            content_column: Column name for backstory content
            novel_column: Column name for novel names
            
        Returns:
            DataFrame with added retrieval results column
        """
        backstories = df[content_column].tolist()
        novel_names = df[novel_column].tolist() if novel_column in df.columns else None
        
        # Batch retrieval
        retrieval_results = self.retrieve_for_backstories_batch(backstories, novel_names)
        
        # Add results to DataFrame
        df_copy = df.copy()
        df_copy['retrieved_chunks'] = retrieval_results
        
        return df_copy
    
    def get_retrieval_stats(self, retrieval_results: List[List[Dict[str, Any]]]) -> Dict[str, float]:
        """
        Get statistics about retrieval results.
        
        Args:
            retrieval_results: List of retrieval result lists
            
        Returns:
            Dictionary with retrieval statistics
        """
        if not retrieval_results:
            return {"avg_chunks": 0.0, "max_chunks": 0, "min_chunks": 0, "total_queries": 0}
        
        chunk_counts = [len(results) for results in retrieval_results]
        
        stats = {
            "avg_chunks": sum(chunk_counts) / len(chunk_counts),
            "max_chunks": max(chunk_counts),
            "min_chunks": min(chunk_counts),
            "total_queries": len(retrieval_results)
        }
        
        # Calculate average similarity scores
        all_similarities = []
        for results in retrieval_results:
            all_similarities.extend([r["similarity_score"] for r in results])
        
        if all_similarities:
            stats["avg_similarity"] = sum(all_similarities) / len(all_similarities)
            stats["max_similarity"] = max(all_similarities)
            stats["min_similarity"] = min(all_similarities)
        
        return stats
