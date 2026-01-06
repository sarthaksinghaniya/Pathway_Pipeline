"""
Simplified Pathway-based novel ingestion and indexing system.
Working implementation that handles Pathway API limitations.
"""

import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import tiktoken
from dataclasses import dataclass


@dataclass
class NovelChunk:
    """Data class representing a novel text chunk with metadata."""
    chunk_id: int
    novel_name: str
    text: str
    token_count: int
    start_char: int
    end_char: int


class SimplePathwayIngester:
    """Handles ingestion of novels from .txt files with Pathway-like interface."""
    
    def __init__(self, novels_dir: str, chunk_size: int = 1000, overlap: int = 150):
        """
        Initialize the novel ingester.
        
        Args:
            novels_dir: Directory containing novel .txt files
            chunk_size: Target chunk size in tokens
            overlap: Token overlap between consecutive chunks
        """
        self.novels_dir = novels_dir
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using cl100k_base encoding."""
        return len(self.encoding.encode(text))
    
    def load_novel_text(self, file_path: str) -> str:
        """Load novel text from file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def chunk_text_by_tokens(self, text: str, novel_name: str) -> List[NovelChunk]:
        """Split text into overlapping chunks based on token count."""
        chunks = []
        chunk_id = 0
        
        # Get token boundaries
        tokens = self.encoding.encode(text)
        total_tokens = len(tokens)
        
        # Calculate chunk boundaries
        chunk_start = 0
        while chunk_start < total_tokens:
            chunk_end = min(chunk_start + self.chunk_size, total_tokens)
            
            # Get chunk tokens and decode
            chunk_tokens = tokens[chunk_start:chunk_end]
            chunk_text = self.encoding.decode(chunk_tokens)
            
            # Calculate character positions (approximation)
            if chunk_start == 0:
                start_char = 0
            else:
                prev_tokens = tokens[:chunk_start]
                start_char = len(self.encoding.decode(prev_tokens))
            
            end_char = start_char + len(chunk_text)
            
            # Create chunk
            chunk = NovelChunk(
                chunk_id=chunk_id,
                novel_name=novel_name,
                text=chunk_text,
                token_count=len(chunk_tokens),
                start_char=start_char,
                end_char=end_char
            )
            chunks.append(chunk)
            
            # Move to next chunk with overlap
            chunk_start = chunk_end - self.overlap
            chunk_id += 1
        
        return chunks
    
    def process_all_novels(self) -> List[NovelChunk]:
        """Process all .txt files in the novels directory."""
        all_chunks = []
        
        if not os.path.exists(self.novels_dir):
            raise FileNotFoundError(f"Novels directory not found: {self.novels_dir}")
        
        for filename in os.listdir(self.novels_dir):
            if filename.endswith('.txt'):
                novel_path = os.path.join(self.novels_dir, filename)
                novel_name = os.path.splitext(filename)[0]
                
                print(f"Processing novel: {novel_name}")
                
                text = self.load_novel_text(novel_path)
                chunks = self.chunk_text_by_tokens(text, novel_name)
                all_chunks.extend(chunks)
                
                print(f"Generated {len(chunks)} chunks from {novel_name}")
        
        return all_chunks
    
    def create_dataframe(self, chunks: List[NovelChunk]) -> pd.DataFrame:
        """Create a pandas DataFrame from novel chunks (Pathway-like table)."""
        records = []
        for chunk in chunks:
            record = {
                "chunk_id": chunk.chunk_id,
                "novel_name": chunk.novel_name,
                "text": chunk.text,
                "token_count": chunk.token_count,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char
            }
            records.append(record)
        
        return pd.DataFrame(records)


class SimpleVectorIndexer:
    """Handles vector indexing of novel chunks using sentence transformers."""
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """Initialize the vector indexer."""
        self.embedding_model = embedding_model
        self._model = None
    
    def _get_model(self):
        """Lazy load the embedding model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.embedding_model)
        return self._model
    
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """Create embeddings for a list of texts."""
        model = self._get_model()
        return model.encode(texts, show_progress_bar=True)
    
    def create_indexed_dataframe(self, chunks_df: pd.DataFrame) -> pd.DataFrame:
        """Create indexed dataframe with embeddings."""
        print("Creating embeddings...")
        embeddings = self.create_embeddings(chunks_df["text"].tolist())
        
        # Add embeddings to dataframe
        indexed_df = chunks_df.copy()
        indexed_df["embedding"] = [emb.tolist() for emb in embeddings]
        
        return indexed_df


class SimpleRetriever:
    """Handles retrieval of relevant chunks using cosine similarity."""
    
    def __init__(self, indexed_df: pd.DataFrame, embedding_model: str = "all-MiniLM-L6-v2"):
        """Initialize the retriever."""
        self.indexed_df = indexed_df
        self.embedding_model = embedding_model
        
        # Pre-compute embedding matrix
        self.embedding_matrix = np.array(indexed_df["embedding"].tolist())
        
        # Load model for queries
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(embedding_model)
    
    def create_query_embedding(self, query: str) -> np.ndarray:
        """Create embedding for query text."""
        return self.model.encode(query)
    
    def cosine_similarity(self, query_embedding: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between query and all documents."""
        # Normalize embeddings
        doc_norms = np.linalg.norm(self.embedding_matrix, axis=1)
        query_norm = np.linalg.norm(query_embedding)
        
        # Compute similarities
        similarities = np.dot(self.embedding_matrix, query_embedding) / (doc_norms * query_norm + 1e-8)
        return similarities
    
    def retrieve_chunks(self, query: str, k: int = 5) -> pd.DataFrame:
        """Retrieve top-K most relevant chunks for a query."""
        # Create query embedding
        query_embedding = self.create_query_embedding(query)
        
        # Compute similarities
        similarities = self.cosine_similarity(query_embedding)
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-k:][::-1]
        
        # Return top-k chunks with similarity scores
        results = self.indexed_df.iloc[top_indices].copy()
        results["similarity"] = similarities[top_indices]
        
        return results.sort_values("similarity", ascending=False)


class SimplePathwayPipeline:
    """Complete pipeline for novel ingestion, indexing, and retrieval."""
    
    def __init__(self, novels_dir: str, chunk_size: int = 1000, overlap: int = 150):
        """Initialize the complete pipeline."""
        self.novels_dir = novels_dir
        self.chunk_size = chunk_size
        self.overlap = overlap
        
        # Initialize components
        self.ingester = SimplePathwayIngester(novels_dir, chunk_size, overlap)
        self.indexer = SimpleVectorIndexer()
        self.retriever = None
        
        # Processed data
        self.chunks_df = None
        self.indexed_df = None
    
    def setup_pipeline(self):
        """Set up the complete pipeline: ingest, index, and prepare for retrieval."""
        print("Setting up novel pipeline...")
        
        # Step 1: Ingest novels
        print("Step 1: Ingesting novels...")
        chunks = self.ingester.process_all_novels()
        self.chunks_df = self.ingester.create_dataframe(chunks)
        print(f"Created dataframe with {len(chunks)} chunks")
        
        # Step 2: Create vector index
        print("Step 2: Creating vector index...")
        self.indexed_df = self.indexer.create_indexed_dataframe(self.chunks_df)
        print("Vector index created")
        
        # Step 3: Initialize retriever
        print("Step 3: Initializing retriever...")
        self.retriever = SimpleRetriever(self.indexed_df)
        print("Pipeline setup complete!")
    
    def retrieve_chunks(self, query: str, k: int = 5) -> pd.DataFrame:
        """Retrieve top-K most relevant chunks for a query."""
        if self.retriever is None:
            raise RuntimeError("Pipeline not set up. Call setup_pipeline() first.")
        
        return self.retriever.retrieve_chunks(query, k)
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get statistics about the pipeline."""
        if self.chunks_df is None:
            return {"status": "not_initialized"}
        
        stats = {
            "total_chunks": len(self.chunks_df),
            "novels": self.chunks_df["novel_name"].nunique(),
            "novel_names": self.chunks_df["novel_name"].unique().tolist(),
            "avg_tokens_per_chunk": self.chunks_df["token_count"].mean(),
            "min_tokens": self.chunks_df["token_count"].min(),
            "max_tokens": self.chunks_df["token_count"].max()
        }
        
        return stats


# Main retrieval function as requested
def retrieve_chunks(query: str, k: int = 5) -> pd.DataFrame:
    """
    Global function to retrieve top-K most relevant chunks.
    
    Args:
        query: Query string
        k: Number of chunks to retrieve
        
    Returns:
        DataFrame with top-K chunks and similarity scores
    """
    # Initialize pipeline (in production, this would be pre-initialized)
    pipeline = SimplePathwayPipeline("data/novel", chunk_size=1000, overlap=150)
    
    # Setup if not already done
    if pipeline.chunks_df is None:
        pipeline.setup_pipeline()
    
    # Retrieve chunks
    return pipeline.retrieve_chunks(query, k)


# Example usage
def main():
    """Example usage of the pipeline."""
    # Initialize pipeline
    pipeline = SimplePathwayPipeline(
        novels_dir="data/novel",
        chunk_size=1000,
        overlap=150
    )
    
    # Set up pipeline
    pipeline.setup_pipeline()
    
    # Get statistics
    stats = pipeline.get_pipeline_stats()
    print("Pipeline Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Example retrieval
    query = "A character who was imprisoned and later escaped"
    results = pipeline.retrieve_chunks(query, k=3)
    
    print(f"\nTop 3 chunks for query: '{query}'")
    for i, (_, row) in enumerate(results.iterrows()):
        print(f"\nChunk {row['chunk_id']} from {row['novel_name']}:")
        print(f"Similarity: {row['similarity']:.3f}")
        print(f"Text preview: {row['text'][:200]}...")
    
    # Test global function
    print(f"\n\nTesting global retrieve_chunks function:")
    global_results = retrieve_chunks("character betrayal revenge", k=2)
    for i, (_, row) in enumerate(global_results.iterrows()):
        print(f"\nGlobal Result {i+1}:")
        print(f"  Novel: {row['novel_name']}")
        print(f"  Similarity: {row['similarity']:.3f}")
        print(f"  Preview: {row['text'][:100]}...")


if __name__ == "__main__":
    main()
