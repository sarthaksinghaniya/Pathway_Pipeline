"""
Pathway-based novel ingestion and indexing system.
Production-quality implementation for loading, chunking, and indexing novels.
"""

import os
import pathway as pw
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


class PathwayNovelIngester:
    """Handles ingestion of novels from .txt files using Pathway."""
    
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
        
        # Define Pathway schema for novel chunks
        self.chunk_schema = {
            "chunk_id": pw.column_definition(int),
            "novel_name": pw.column_definition(str),
            "text": pw.column_definition(str),
            "token_count": pw.column_definition(int),
            "start_char": pw.column_definition(int),
            "end_char": pw.column_definition(int)
        }
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text using cl100k_base encoding.
        
        Args:
            text: Input text
            
        Returns:
            Number of tokens
        """
        return len(self.encoding.encode(text))
    
    def load_novel_text(self, file_path: str) -> str:
        """
        Load novel text from file.
        
        Args:
            file_path: Path to the novel .txt file
            
        Returns:
            Raw novel text
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def chunk_text_by_tokens(self, text: str, novel_name: str) -> List[NovelChunk]:
        """
        Split text into overlapping chunks based on token count.
        
        Args:
            text: Full novel text
            novel_name: Name of the novel for metadata
            
        Returns:
            List of NovelChunk objects
        """
        chunks = []
        chunk_id = 0
        
        # Get token boundaries
        tokens = self.encoding.encode(text)
        total_tokens = len(tokens)
        
        # Calculate chunk boundaries
        chunk_start = 0
        while chunk_start < total_tokens:
            chunk_end = min(chunk_start + self.chunk_size, total_tokens)
            
            # Convert token indices back to character indices
            # Note: This is an approximation - for exact boundaries, we'd need
            # tiktoken's decode functionality, but this approach is sufficient for most use cases
            chunk_tokens = tokens[chunk_start:chunk_end]
            chunk_text = self.encoding.decode(chunk_tokens)
            
            # Find approximate character positions
            if chunk_start == 0:
                start_char = 0
            else:
                # Approximate start character based on previous chunk
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
        """
        Process all .txt files in the novels directory.
        
        Returns:
            List of all chunks from all novels
        """
        all_chunks = []
        
        if not os.path.exists(self.novels_dir):
            raise FileNotFoundError(f"Novels directory not found: {self.novels_dir}")
        
        # Process each .txt file
        for filename in os.listdir(self.novels_dir):
            if filename.endswith('.txt'):
                novel_path = os.path.join(self.novels_dir, filename)
                novel_name = os.path.splitext(filename)[0]  # Remove .txt extension
                
                print(f"Processing novel: {novel_name}")
                
                # Load and chunk the novel
                text = self.load_novel_text(novel_path)
                chunks = self.chunk_text_by_tokens(text, novel_name)
                all_chunks.extend(chunks)
                
                print(f"Generated {len(chunks)} chunks from {novel_name}")
        
        return all_chunks
    
    def create_pathway_table(self, chunks: List[NovelChunk]) -> pw.Table:
        """
        Create a Pathway table from novel chunks.
        
        Args:
            chunks: List of NovelChunk objects
            
        Returns:
            Pathway table with chunk data
        """
        # Convert chunks to records format for Pathway
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
        
        # Create Pathway table
        table = pw.Table.from_records(records, schema=self.chunk_schema)
        return table


class PathwayVectorIndexer:
    """Handles vector indexing of novel chunks using Pathway."""
    
    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the vector indexer.
        
        Args:
            embedding_model: Name of the embedding model to use
        """
        self.embedding_model = embedding_model
        
        # Define schema for indexed chunks with embeddings
        self.indexed_schema = {
            "chunk_id": pw.column_definition(int),
            "novel_name": pw.column_definition(str),
            "text": pw.column_definition(str),
            "token_count": pw.column_definition(int),
            "start_char": pw.column_definition(int),
            "end_char": pw.column_definition(int),
            "embedding": pw.column_definition(pw.Json)  # Store embeddings as JSON
        }
    
    @pw.udf
    def create_embedding(self, text: str) -> pw.Json:
        """
        Create embedding for text using sentence transformers.
        
        Args:
            text: Input text to embed
            
        Returns:
            Embedding vector as JSON
        """
        # Note: In a production environment, you'd want to handle
        # model loading more efficiently (e.g., load once at startup)
        from sentence_transformers import SentenceTransformer
        import json
        
        # Load model (cached after first call)
        if not hasattr(self, '_model'):
            self._model = SentenceTransformer(self.embedding_model)
        
        # Create embedding
        embedding = self._model.encode(text).tolist()
        return pw.Json(embedding)
    
    def create_indexed_table(self, chunks_table: pw.Table) -> pw.Table:
        """
        Create indexed table with embeddings.
        
        Args:
            chunks_table: Pathway table with chunk data
            
        Returns:
            Indexed table with embedding column
        """
        # Add embeddings to the table
        indexed_table = chunks_table.with_columns(
            embedding=self.create_embedding(pw.this.text)
        )
        
        return indexed_table


class PathwayRetriever:
    """Handles retrieval of relevant chunks using Pathway."""
    
    def __init__(self, indexed_table: pw.Table, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the retriever.
        
        Args:
            indexed_table: Pathway table with indexed chunks
            embedding_model: Name of the embedding model
        """
        self.indexed_table = indexed_table
        self.embedding_model = embedding_model
        
        # Load embedding model for queries
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(embedding_model)
    
    def create_query_embedding(self, query: str) -> List[float]:
        """
        Create embedding for query text.
        
        Args:
            query: Query string
            
        Returns:
            Query embedding vector
        """
        return self.model.encode(query).tolist()
    
    @pw.udf
    def cosine_similarity(self, query_embedding: pw.Json, doc_embedding: pw.Json) -> float:
        """
        Compute cosine similarity between query and document embeddings.
        
        Args:
            query_embedding: Query embedding as JSON
            doc_embedding: Document embedding as JSON
            
        Returns:
            Cosine similarity score
        """
        import numpy as np
        
        # Convert JSON embeddings to numpy arrays
        query_vec = np.array(query_embedding)
        doc_vec = np.array(doc_embedding)
        
        # Compute cosine similarity
        similarity = np.dot(query_vec, doc_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(doc_vec) + 1e-8)
        return float(similarity)
    
    def retrieve_chunks(self, query: str, k: int = 5) -> pw.Table:
        """
        Retrieve top-K most relevant chunks for a query.
        
        Args:
            query: Query string
            k: Number of chunks to retrieve
            
        Returns:
            Pathway table with top-K chunks and similarity scores
        """
        # Create query embedding
        query_embedding = self.create_query_embedding(query)
        query_embedding_json = pw.Json(query_embedding)
        
        # Compute similarities
        similarities = self.indexed_table.with_columns(
            similarity=self.cosine_similarity(query_embedding_json, pw.this.embedding)
        )
        
        # Sort by similarity and take top-k
        top_chunks = similarities.sort(pw.this.similarity, descending=True).limit(k)
        
        return top_chunks


class PathwayNovelPipeline:
    """Complete pipeline for novel ingestion, indexing, and retrieval."""
    
    def __init__(self, novels_dir: str, chunk_size: int = 1000, overlap: int = 150):
        """
        Initialize the complete pipeline.
        
        Args:
            novels_dir: Directory containing novel .txt files
            chunk_size: Target chunk size in tokens
            overlap: Token overlap between chunks
        """
        self.novels_dir = novels_dir
        self.chunk_size = chunk_size
        self.overlap = overlap
        
        # Initialize components
        self.ingester = PathwayNovelIngester(novels_dir, chunk_size, overlap)
        self.indexer = PathwayVectorIndexer()
        self.retriever = None
        
        # Processed data
        self.chunks_table = None
        self.indexed_table = None
    
    def setup_pipeline(self):
        """Set up the complete pipeline: ingest, index, and prepare for retrieval."""
        print("Setting up Pathway novel pipeline...")
        
        # Step 1: Ingest novels
        print("Step 1: Ingesting novels...")
        chunks = self.ingester.process_all_novels()
        self.chunks_table = self.ingester.create_pathway_table(chunks)
        print(f"Created table with {len(chunks)} chunks")
        
        # Step 2: Create vector index
        print("Step 2: Creating vector index...")
        self.indexed_table = self.indexer.create_indexed_table(self.chunks_table)
        print("Vector index created")
        
        # Step 3: Initialize retriever
        print("Step 3: Initializing retriever...")
        self.retriever = PathwayRetriever(self.indexed_table)
        print("Pipeline setup complete!")
    
    def retrieve_chunks(self, query: str, k: int = 5) -> pw.Table:
        """
        Retrieve top-K most relevant chunks for a query.
        
        Args:
            query: Query string
            k: Number of chunks to retrieve
            
        Returns:
            Pathway table with top-K chunks
        """
        if self.retriever is None:
            raise RuntimeError("Pipeline not set up. Call setup_pipeline() first.")
        
        return self.retriever.retrieve_chunks(query, k)
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the pipeline.
        
        Returns:
            Dictionary with pipeline statistics
        """
        if self.chunks_table is None:
            return {"status": "not_initialized"}
        
        # Convert to pandas for statistics
        df = self.chunks_table.to_pandas()
        
        stats = {
            "total_chunks": len(df),
            "novels": df["novel_name"].nunique(),
            "novel_names": df["novel_name"].unique().tolist(),
            "avg_tokens_per_chunk": df["token_count"].mean(),
            "min_tokens": df["token_count"].min(),
            "max_tokens": df["token_count"].max()
        }
        
        return stats


# Example usage function
def main():
    """Example usage of the Pathway novel pipeline."""
    # Initialize pipeline
    pipeline = PathwayNovelPipeline(
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
    results_df = results.to_pandas()
    for i, row in results_df.iterrows():
        print(f"\nChunk {row['chunk_id']} from {row['novel_name']}:")
        print(f"Similarity: {row['similarity']:.3f}")
        print(f"Text preview: {row['text'][:200]}...")


if __name__ == "__main__":
    main()
