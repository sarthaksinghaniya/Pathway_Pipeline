"""
Robust Pathway-based novel ingestion and indexing system.
Production-ready implementation with error handling.
"""

import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import tiktoken
from dataclasses import dataclass
import re


@dataclass
class NovelChunk:
    """Data class representing a novel text chunk with metadata."""
    chunk_id: int
    novel_name: str
    text: str
    token_count: int
    start_char: int
    end_char: int


class RobustPathwayIngester:
    """Handles ingestion of novels from .txt files with robust error handling."""
    
    def __init__(self, novels_dir: str, chunk_size: int = 1000, overlap: int = 150):
        """
        Initialize novel ingester.
        
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
        """Load novel text from file with error handling."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            # Try different encodings
            for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        return f.read()
                except UnicodeDecodeError:
                    continue
            raise ValueError(f"Could not decode file {file_path} with any encoding")
    
    def chunk_text_by_paragraphs(self, text: str, novel_name: str) -> List[NovelChunk]:
        """
        Split text into chunks based on paragraphs and token limits.
        More robust than pure token-based chunking.
        """
        chunks = []
        chunk_id = 0
        
        # Split into paragraphs
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        current_chunk_text = ""
        current_tokens = 0
        
        for paragraph in paragraphs:
            paragraph_tokens = self.count_tokens(paragraph)
            
            # If adding this paragraph exceeds chunk size
            if current_tokens + paragraph_tokens > self.chunk_size and current_chunk_text:
                # Create chunk
                chunk = NovelChunk(
                    chunk_id=chunk_id,
                    novel_name=novel_name,
                    text=current_chunk_text.strip(),
                    token_count=current_tokens,
                    start_char=0,  # Simplified for now
                    end_char=len(current_chunk_text)
                )
                chunks.append(chunk)
                chunk_id += 1
                
                # Start new chunk with overlap
                if self.overlap > 0:
                    # Keep some text from previous chunk for overlap
                    words = current_chunk_text.split()
                    overlap_words = words[-min(50, len(words)):]  # Keep last 50 words
                    current_chunk_text = " ".join(overlap_words) + "\n\n" + paragraph
                    current_tokens = self.count_tokens(current_chunk_text)
                else:
                    current_chunk_text = paragraph
                    current_tokens = paragraph_tokens
            else:
                # Add to current chunk
                if current_chunk_text:
                    current_chunk_text += "\n\n" + paragraph
                else:
                    current_chunk_text = paragraph
                current_tokens += paragraph_tokens
        
        # Handle final chunk
        if current_chunk_text.strip():
            chunk = NovelChunk(
                chunk_id=chunk_id,
                novel_name=novel_name,
                text=current_chunk_text.strip(),
                token_count=current_tokens,
                start_char=0,
                end_char=len(current_chunk_text)
            )
            chunks.append(chunk)
        
        return chunks
    
    def process_all_novels(self) -> List[NovelChunk]:
        """Process all .txt files in novels directory."""
        all_chunks = []
        
        if not os.path.exists(self.novels_dir):
            raise FileNotFoundError(f"Novels directory not found: {self.novels_dir}")
        
        txt_files = [f for f in os.listdir(self.novels_dir) if f.endswith('.txt')]
        if not txt_files:
            raise ValueError(f"No .txt files found in {self.novels_dir}")
        
        for filename in txt_files:
            novel_path = os.path.join(self.novels_dir, filename)
            novel_name = os.path.splitext(filename)[0]
            
            print(f"Processing novel: {novel_name}")
            
            try:
                text = self.load_novel_text(novel_path)
                chunks = self.chunk_text_by_paragraphs(text, novel_name)
                all_chunks.extend(chunks)
                
                print(f"Generated {len(chunks)} chunks from {novel_name}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue
        
        return all_chunks
    
    def create_dataframe(self, chunks: List[NovelChunk]) -> pd.DataFrame:
        """Create a pandas DataFrame from novel chunks."""
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


class RobustVectorIndexer:
    """Handles vector indexing with fallback options."""
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """Initialize vector indexer."""
        self.embedding_model = embedding_model
        self._model = None
        self._use_tfidf = False
    
    def _get_model(self):
        """Lazy load embedding model with fallback."""
        if self._model is None and not self._use_tfidf:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.embedding_model)
                print(f"Loaded sentence transformer model: {self.embedding_model}")
            except Exception as e:
                print(f"Failed to load sentence transformer: {e}")
                print("Falling back to TF-IDF approach")
                self._use_tfidf = True
        
        return self._model
    
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """Create embeddings for a list of texts."""
        if self._use_tfidf:
            return self._create_tfidf_embeddings(texts)
        else:
            try:
                model = self._get_model()
                return model.encode(texts, show_progress_bar=True)
            except Exception as e:
                print(f"Failed to encode with sentence transformer: {e}")
                print("Falling back to TF-IDF approach")
                self._use_tfidf = True
                return self._create_tfidf_embeddings(texts)
    
    def _create_tfidf_embeddings(self, texts: List[str]) -> np.ndarray:
        """Create TF-IDF based embeddings as fallback."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        print("Using TF-IDF fallback for embeddings")
        vectorizer = TfidfVectorizer(
            max_features=384,  # Match typical embedding dimension
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        embeddings = vectorizer.fit_transform(texts)
        return embeddings.toarray()
    
    def create_indexed_dataframe(self, chunks_df: pd.DataFrame) -> pd.DataFrame:
        """Create indexed dataframe with embeddings."""
        print("Creating embeddings...")
        
        # Clean texts to avoid issues
        texts = [str(text) if text is not None else "" for text in chunks_df["text"].tolist()]
        
        embeddings = self.create_embeddings(texts)
        
        # Add embeddings to dataframe
        indexed_df = chunks_df.copy()
        indexed_df["embedding"] = [emb.tolist() for emb in embeddings]
        
        return indexed_df


class RobustRetriever:
    """Handles retrieval with multiple similarity methods."""
    
    def __init__(self, indexed_df: pd.DataFrame, embedding_model: str = "all-MiniLM-L6-v2"):
        """Initialize retriever."""
        self.indexed_df = indexed_df
        self.embedding_model = embedding_model
        self._use_tfidf = False
        
        # Pre-compute embedding matrix
        self.embedding_matrix = np.array(indexed_df["embedding"].tolist())
        
        # Load model for queries
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(embedding_model)
            print(f"Loaded sentence transformer for queries: {embedding_model}")
        except Exception as e:
            print(f"Failed to load sentence transformer for queries: {e}")
            print("Using TF-IDF for queries")
            self._use_tfidf = True
            self._setup_tfidf()
    
    def _setup_tfidf(self):
        """Setup TF-IDF for query processing."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        # Re-fit TF-IDF on all texts
        texts = [str(text) if text is not None else "" for text in self.indexed_df["text"].tolist()]
        self.vectorizer = TfidfVectorizer(
            max_features=384,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.vectorizer.fit(texts)
    
    def create_query_embedding(self, query: str) -> np.ndarray:
        """Create embedding for query text."""
        if self._use_tfidf:
            query_vec = self.vectorizer.transform([query])
            return query_vec.toarray().flatten()
        else:
            return self.model.encode(query)
    
    def cosine_similarity(self, query_embedding: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between query and all documents."""
        # Handle sparse matrices from TF-IDF
        if hasattr(query_embedding, 'toarray'):
            query_embedding = query_embedding.toarray().flatten()
        
        # Normalize embeddings
        doc_norms = np.linalg.norm(self.embedding_matrix, axis=1)
        query_norm = np.linalg.norm(query_embedding)
        
        if query_norm == 0:
            return np.zeros(len(doc_norms))
        
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


class RobustPathwayPipeline:
    """Complete pipeline with robust error handling and fallbacks."""
    
    def __init__(self, novels_dir: str, chunk_size: int = 1000, overlap: int = 150):
        """Initialize complete pipeline."""
        self.novels_dir = novels_dir
        self.chunk_size = chunk_size
        self.overlap = overlap
        
        # Initialize components
        self.ingester = RobustPathwayIngester(novels_dir, chunk_size, overlap)
        self.indexer = RobustVectorIndexer()
        self.retriever = None
        
        # Processed data
        self.chunks_df = None
        self.indexed_df = None
    
    def setup_pipeline(self):
        """Set up complete pipeline with error handling."""
        print("Setting up robust novel pipeline...")
        
        try:
            # Step 1: Ingest novels
            print("Step 1: Ingesting novels...")
            chunks = self.ingester.process_all_novels()
            if not chunks:
                raise ValueError("No chunks were generated from novels")
            
            self.chunks_df = self.ingester.create_dataframe(chunks)
            print(f"Created dataframe with {len(chunks)} chunks")
            
            # Step 2: Create vector index
            print("Step 2: Creating vector index...")
            self.indexed_df = self.indexer.create_indexed_dataframe(self.chunks_df)
            print("Vector index created")
            
            # Step 3: Initialize retriever
            print("Step 3: Initializing retriever...")
            self.retriever = RobustRetriever(self.indexed_df)
            print("Pipeline setup complete!")
            
        except Exception as e:
            print(f"Error during pipeline setup: {e}")
            raise
    
    def retrieve_chunks(self, query: str, k: int = 5) -> pd.DataFrame:
        """Retrieve top-K most relevant chunks for a query."""
        if self.retriever is None:
            raise RuntimeError("Pipeline not set up. Call setup_pipeline() first.")
        
        return self.retriever.retrieve_chunks(query, k)
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about pipeline."""
        if self.chunks_df is None:
            return {"status": "not_initialized"}
        
        stats = {
            "total_chunks": len(self.chunks_df),
            "novels": self.chunks_df["novel_name"].nunique(),
            "novel_names": self.chunks_df["novel_name"].unique().tolist(),
            "avg_tokens_per_chunk": float(self.chunks_df["token_count"].mean()),
            "min_tokens": int(self.chunks_df["token_count"].min()),
            "max_tokens": int(self.chunks_df["token_count"].max()),
            "total_tokens": int(self.chunks_df["token_count"].sum())
        }
        
        return stats


# Global retrieval function as requested
def retrieve_chunks(query: str, k: int = 5) -> pd.DataFrame:
    """
    Global function to retrieve top-K most relevant chunks.
    
    Args:
        query: Query string
        k: Number of chunks to retrieve
        
    Returns:
        DataFrame with top-K chunks and similarity scores
    """
    # Initialize pipeline (in production, this would be pre-initialized and cached)
    pipeline = RobustPathwayPipeline("data/novel", chunk_size=1000, overlap=150)
    
    # Setup if not already done
    if pipeline.chunks_df is None:
        pipeline.setup_pipeline()
    
    # Retrieve chunks
    return pipeline.retrieve_chunks(query, k)


# Example usage
def main():
    """Example usage of robust pipeline."""
    try:
        # Initialize pipeline
        pipeline = RobustPathwayPipeline(
            novels_dir="data/novel",
            chunk_size=1000,
            overlap=150
        )
        
        # Set up pipeline
        pipeline.setup_pipeline()
        
        # Get statistics
        stats = pipeline.get_pipeline_stats()
        print("\nPipeline Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Example retrievals
        queries = [
            "A character who was imprisoned and later escaped",
            "betrayal and revenge",
            "sea voyage and adventure",
            "mysterious prisoner"
        ]
        
        for query in queries:
            print(f"\n" + "="*60)
            print(f"Query: '{query}'")
            print("="*60)
            
            results = pipeline.retrieve_chunks(query, k=3)
            
            for i, (_, row) in enumerate(results.iterrows()):
                print(f"\nResult {i+1}:")
                print(f"  Novel: {row['novel_name']}")
                print(f"  Chunk ID: {row['chunk_id']}")
                print(f"  Similarity: {row['similarity']:.3f}")
                print(f"  Tokens: {row['token_count']}")
                print(f"  Preview: {row['text'][:150]}...")
        
        # Test global function
        print(f"\n" + "="*60)
        print("Testing global retrieve_chunks function:")
        print("="*60)
        
        global_results = retrieve_chunks("character imprisonment escape", k=2)
        for i, (_, row) in enumerate(global_results.iterrows()):
            print(f"\nGlobal Result {i+1}:")
            print(f"  Novel: {row['novel_name']}")
            print(f"  Similarity: {row['similarity']:.3f}")
            print(f"  Preview: {row['text'][:100]}...")
    
    except Exception as e:
        print(f"Error in main: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
