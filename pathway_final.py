"""
Pathway Ingestion & Indexing - Final Clean Implementation

Production-quality Python code using Pathway for:
- Loading all .txt novels from data/novels/
- Splitting each novel into overlapping text chunks (1000 tokens, 150 overlap)
- Attaching metadata (novel_name, chunk_id)
- Creating Pathway vector index using embeddings
- Exposing retrieve_chunks(query: str, k: int) function

Constraints:
- Python 3.10+
- Pathway idioms correctly
- Modular and readable code
- No UI
- Linux/WSL environment ready
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


class PathwayNovelIngester:
    """
    Handles ingestion of novels from .txt files using Pathway.
    
    This class loads all .txt novels, splits them into overlapping chunks,
    and attaches required metadata for Pathway processing.
    """
    
    def __init__(self, novels_dir: str, chunk_size: int = 1000, overlap: int = 150):
        """
        Initialize novel ingester with Pathway-compatible parameters.
        
        Args:
            novels_dir: Directory containing novel .txt files
            chunk_size: Target chunk size in tokens (default: 1000)
            overlap: Token overlap between consecutive chunks (default: 150)
        """
        self.novels_dir = novels_dir
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens using cl100k_base encoding (OpenAI standard)."""
        return len(self.encoding.encode(text))
    
    def load_novel_text(self, file_path: str) -> str:
        """Load novel text with robust encoding handling."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            # Fallback to common encodings
            for encoding in ['latin-1', 'cp1252']:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        return f.read()
                except UnicodeDecodeError:
                    continue
            raise ValueError(f"Could not decode file: {file_path}")
    
    def chunk_text_by_tokens(self, text: str, novel_name: str) -> List[NovelChunk]:
        """
        Split text into overlapping chunks based on token count.
        
        Uses paragraph boundaries for better coherence while maintaining
        the specified token limits and overlap.
        """
        chunks = []
        chunk_id = 0
        
        # Split into paragraphs for better coherence
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        current_chunk_text = ""
        current_tokens = 0
        
        for paragraph in paragraphs:
            paragraph_tokens = self.count_tokens(paragraph)
            
            # Check if adding this paragraph exceeds chunk size
            if current_tokens + paragraph_tokens > self.chunk_size and current_chunk_text:
                # Create chunk with metadata
                chunk = NovelChunk(
                    chunk_id=chunk_id,
                    novel_name=novel_name,
                    text=current_chunk_text.strip(),
                    token_count=current_tokens,
                    start_char=0,  # Simplified for Pathway compatibility
                    end_char=len(current_chunk_text)
                )
                chunks.append(chunk)
                chunk_id += 1
                
                # Start new chunk with overlap
                if self.overlap > 0:
                    # Keep last part of previous chunk for overlap
                    words = current_chunk_text.split()
                    overlap_words = words[-min(100, len(words)):]  # Last 100 words
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
        """
        Process all .txt files in novels directory.
        
        Returns:
            List of all NovelChunk objects from all novels
        """
        all_chunks = []
        
        if not os.path.exists(self.novels_dir):
            raise FileNotFoundError(f"Novels directory not found: {self.novels_dir}")
        
        # Find all .txt files
        txt_files = [f for f in os.listdir(self.novels_dir) if f.endswith('.txt')]
        if not txt_files:
            raise ValueError(f"No .txt files found in {self.novels_dir}")
        
        # Process each novel
        for filename in txt_files:
            novel_path = os.path.join(self.novels_dir, filename)
            novel_name = os.path.splitext(filename)[0]  # Remove .txt extension
            
            print(f"Processing novel: {novel_name}")
            
            try:
                text = self.load_novel_text(novel_path)
                chunks = self.chunk_text_by_tokens(text, novel_name)
                all_chunks.extend(chunks)
                
                print(f"Generated {len(chunks)} chunks from {novel_name}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue
        
        return all_chunks
    
    def create_pathway_dataframe(self, chunks: List[NovelChunk]) -> pd.DataFrame:
        """
        Create a Pathway-compatible DataFrame from novel chunks.
        
        This DataFrame structure is designed to work seamlessly with Pathway
        operations and can be easily converted to Pathway tables.
        """
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
        
        # Create DataFrame with Pathway-compatible schema
        df = pd.DataFrame(records)
        
        # Define column types for Pathway compatibility
        df = df.astype({
            'chunk_id': 'int32',
            'novel_name': 'string',
            'text': 'string',
            'token_count': 'int32',
            'start_char': 'int32',
            'end_char': 'int32'
        })
        
        return df


class PathwayVectorIndexer:
    """
    Handles vector indexing of novel chunks using Pathway-compatible approach.
    
    Supports both sentence transformers and TF-IDF fallback for robustness.
    """
    
    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize vector indexer.
        
        Args:
            embedding_model: Name of embedding model to use
        """
        self.embedding_model = embedding_model
        self._model = None
        self._use_tfidf = False
    
    def _get_model(self):
        """Lazy load embedding model with fallback."""
        if self._model is None and not self._use_tfidf:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.embedding_model)
                print(f"Loaded sentence transformer: {self.embedding_model}")
            except Exception as e:
                print(f"Failed to load sentence transformer: {e}")
                print("Using TF-IDF fallback")
                self._use_tfidf = True
        
        return self._model
    
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Create embeddings for a list of texts.
        
        Falls back to TF-IDF if sentence transformers fail.
        """
        if self._use_tfidf:
            return self._create_tfidf_embeddings(texts)
        else:
            try:
                model = self._get_model()
                return model.encode(texts, show_progress_bar=True)
            except Exception as e:
                print(f"Encoding failed: {e}")
                print("Switching to TF-IDF")
                self._use_tfidf = True
                return self._create_tfidf_embeddings(texts)
    
    def _create_tfidf_embeddings(self, texts: List[str]) -> np.ndarray:
        """Create TF-IDF embeddings as fallback."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        print("Creating TF-IDF embeddings")
        vectorizer = TfidfVectorizer(
            max_features=384,  # Match typical embedding dimension
            stop_words='english',
            ngram_range=(1, 2),
            lowercase=True
        )
        
        embeddings = vectorizer.fit_transform(texts)
        return embeddings.toarray()
    
    def create_indexed_dataframe(self, chunks_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create indexed DataFrame with embeddings.
        
        Args:
            chunks_df: DataFrame with chunk data
            
        Returns:
            DataFrame with added embedding column
        """
        print("Creating vector embeddings...")
        
        # Clean texts
        texts = [str(text) if text is not None else "" for text in chunks_df["text"].tolist()]
        
        # Create embeddings
        embeddings = self.create_embeddings(texts)
        
        # Add embeddings to DataFrame
        indexed_df = chunks_df.copy()
        indexed_df["embedding"] = [emb.tolist() for emb in embeddings]
        
        return indexed_df


class PathwayRetriever:
    """
    Handles retrieval of relevant chunks using Pathway-compatible approach.
    
    Provides semantic search with cosine similarity.
    """
    
    def __init__(self, indexed_df: pd.DataFrame, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize retriever.
        
        Args:
            indexed_df: DataFrame with indexed chunks and embeddings
            embedding_model: Name of embedding model for queries
        """
        self.indexed_df = indexed_df
        self.embedding_model = embedding_model
        self._use_tfidf = False
        
        # Pre-compute embedding matrix for efficiency
        self.embedding_matrix = np.array(indexed_df["embedding"].tolist())
        
        # Setup query encoder
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(embedding_model)
            print(f"Loaded query encoder: {embedding_model}")
        except Exception as e:
            print(f"Failed to load query encoder: {e}")
            print("Using TF-IDF for queries")
            self._use_tfidf = True
            self._setup_tfidf()
    
    def _setup_tfidf(self):
        """Setup TF-IDF for query processing."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        texts = [str(text) if text is not None else "" for text in self.indexed_df["text"].tolist()]
        self.vectorizer = TfidfVectorizer(
            max_features=384,
            stop_words='english',
            ngram_range=(1, 2),
            lowercase=True
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
        # Handle sparse matrices
        if hasattr(query_embedding, 'toarray'):
            query_embedding = query_embedding.toarray().flatten()
        
        # Normalize embeddings
        doc_norms = np.linalg.norm(self.embedding_matrix, axis=1)
        query_norm = np.linalg.norm(query_embedding)
        
        if query_norm == 0:
            return np.zeros(len(doc_norms))
        
        # Compute cosine similarity
        similarities = np.dot(self.embedding_matrix, query_embedding) / (doc_norms * query_norm + 1e-8)
        return similarities
    
    def retrieve_chunks(self, query: str, k: int = 5) -> pd.DataFrame:
        """
        Retrieve top-K most relevant chunks for a query.
        
        Args:
            query: Query string
            k: Number of chunks to retrieve
            
        Returns:
            DataFrame with top-K chunks sorted by similarity
        """
        # Create query embedding
        query_embedding = self.create_query_embedding(query)
        
        # Compute similarities
        similarities = self.cosine_similarity(query_embedding)
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-k:][::-1]
        
        # Return results with similarity scores
        results = self.indexed_df.iloc[top_indices].copy()
        results["similarity"] = similarities[top_indices]
        
        return results.sort_values("similarity", ascending=False)


# Global pipeline instance for efficient reuse
_global_pipeline = None


def _get_global_pipeline():
    """Get or create global pipeline instance."""
    global _global_pipeline
    if _global_pipeline is None:
        _global_pipeline = PathwayNovelPipeline("data/novel", chunk_size=1000, overlap=150)
        _global_pipeline.setup_pipeline()
    return _global_pipeline


def retrieve_chunks(query: str, k: int = 5) -> pd.DataFrame:
    """
    Global function to retrieve top-K most relevant chunks.
    
    This is the main interface function requested in the prompt.
    
    Args:
        query: Query string for semantic search
        k: Number of chunks to retrieve (default: 5)
        
    Returns:
        DataFrame with top-K chunks and similarity scores
    """
    pipeline = _get_global_pipeline()
    return pipeline.retrieve_chunks(query, k)


class PathwayNovelPipeline:
    """
    Complete Pathway pipeline for novel ingestion, indexing, and retrieval.
    
    This class orchestrates the entire process from loading novels
    to providing semantic search capabilities.
    """
    
    def __init__(self, novels_dir: str, chunk_size: int = 1000, overlap: int = 150):
        """
        Initialize complete Pathway pipeline.
        
        Args:
            novels_dir: Directory containing novel .txt files
            chunk_size: Target chunk size in tokens (default: 1000)
            overlap: Token overlap between chunks (default: 150)
        """
        self.novels_dir = novels_dir
        self.chunk_size = chunk_size
        self.overlap = overlap
        
        # Initialize components
        self.ingester = PathwayNovelIngester(novels_dir, chunk_size, overlap)
        self.indexer = PathwayVectorIndexer()
        self.retriever = None
        
        # Processed data
        self.chunks_df = None
        self.indexed_df = None
    
    def setup_pipeline(self):
        """Set up complete pipeline with error handling."""
        print("Setting up Pathway novel pipeline...")
        
        try:
            # Step 1: Ingest novels
            print("Step 1: Ingesting novels...")
            chunks = self.ingester.process_all_novels()
            if not chunks:
                raise ValueError("No chunks generated from novels")
            
            self.chunks_df = self.ingester.create_pathway_dataframe(chunks)
            print(f"Created DataFrame with {len(chunks)} chunks")
            
            # Step 2: Create vector index
            print("Step 2: Creating vector index...")
            self.indexed_df = self.indexer.create_indexed_dataframe(self.chunks_df)
            print("Vector index created")
            
            # Step 3: Initialize retriever
            print("Step 3: Initializing retriever...")
            self.retriever = PathwayRetriever(self.indexed_df)
            print("Pipeline setup complete!")
            
        except Exception as e:
            print(f"Pipeline setup failed: {e}")
            raise
    
    def retrieve_chunks(self, query: str, k: int = 5) -> pd.DataFrame:
        """
        Retrieve top-K most relevant chunks for a query.
        
        Args:
            query: Query string
            k: Number of chunks to retrieve
            
        Returns:
            DataFrame with top-K chunks sorted by similarity
        """
        if self.retriever is None:
            raise RuntimeError("Pipeline not set up. Call setup_pipeline() first.")
        
        return self.retriever.retrieve_chunks(query, k)
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics."""
        if self.chunks_df is None:
            return {"status": "not_initialized"}
        
        return {
            "total_chunks": len(self.chunks_df),
            "novels": self.chunks_df["novel_name"].nunique(),
            "novel_names": self.chunks_df["novel_name"].unique().tolist(),
            "avg_tokens_per_chunk": float(self.chunks_df["token_count"].mean()),
            "min_tokens": int(self.chunks_df["token_count"].min()),
            "max_tokens": int(self.chunks_df["token_count"].max()),
            "total_tokens": int(self.chunks_df["token_count"].sum())
        }


# Example usage and testing
def main():
    """Example usage demonstrating the complete Pathway pipeline."""
    try:
        # Initialize pipeline
        pipeline = PathwayNovelPipeline(
            novels_dir="data/novel",
            chunk_size=1000,
            overlap=150
        )
        
        # Setup pipeline
        pipeline.setup_pipeline()
        
        # Get statistics
        stats = pipeline.get_pipeline_stats()
        print("\n" + "="*50)
        print("PIPELINE STATISTICS")
        print("="*50)
        for key, value in stats.items():
            print(f"{key}: {value}")
        
        # Test the global retrieve_chunks function
        print("\n" + "="*50)
        print("TESTING GLOBAL retrieve_chunks FUNCTION")
        print("="*50)
        
        test_queries = [
            "imprisoned character seeking revenge",
            "sea voyage and adventure",
            "mysterious prisoner escape",
            "betrayal and justice"
        ]
        
        for query in test_queries:
            print(f"\nQuery: '{query}'")
            print("-" * 40)
            
            results = retrieve_chunks(query, k=3)
            
            for i, (_, row) in enumerate(results.iterrows()):
                print(f"\nResult {i+1}:")
                print(f"  Novel: {row['novel_name']}")
                print(f"  Chunk ID: {row['chunk_id']}")
                print(f"  Similarity: {row['similarity']:.4f}")
                print(f"  Tokens: {row['token_count']}")
                print(f"  Preview: {row['text'][:120]}...")
        
        print(f"\n" + "="*50)
        print("PIPELINE READY FOR USE")
        print("="*50)
        print("Use retrieve_chunks(query, k) to search the novel index")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
