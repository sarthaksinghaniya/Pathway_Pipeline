"""
Novel ingestion and text chunking module.
Handles loading novels from .txt files and chunking them into overlapping segments.
"""

import os
import tiktoken
from typing import List, Dict, Any
import pathway as pw


class NovelChunker:
    """Handles novel ingestion and chunking into overlapping segments."""
    
    def __init__(self, min_tokens: int = 800, max_tokens: int = 1200, overlap: int = 200):
        """
        Initialize the chunker with token parameters.
        
        Args:
            min_tokens: Minimum tokens per chunk
            max_tokens: Maximum tokens per chunk  
            overlap: Number of overlapping tokens between chunks
        """
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.overlap = overlap
        self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using cl100k_base encoding."""
        return len(self.encoding.encode(text))
    
    def load_novel(self, file_path: str) -> str:
        """
        Load novel text from file.
        
        Args:
            file_path: Path to the novel .txt file
            
        Returns:
            Raw novel text
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def chunk_text(self, text: str, novel_name: str) -> List[Dict[str, Any]]:
        """
        Chunk text into overlapping segments within token limits.
        
        Args:
            text: Full novel text
            novel_name: Name of the novel for metadata
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        # Split into paragraphs first to maintain coherence
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        chunks = []
        current_chunk = ""
        chunk_id = 0
        
        for paragraph in paragraphs:
            test_chunk = current_chunk + "\n\n" + paragraph if current_chunk else paragraph
            
            if self.count_tokens(test_chunk) <= self.max_tokens:
                current_chunk = test_chunk
            else:
                # Save current chunk if it meets minimum requirements
                if self.count_tokens(current_chunk) >= self.min_tokens:
                    chunks.append({
                        "chunk_id": chunk_id,
                        "novel_name": novel_name,
                        "text": current_chunk,
                        "token_count": self.count_tokens(current_chunk)
                    })
                    chunk_id += 1
                
                # Start new chunk with overlap
                if current_chunk:
                    # Create overlap by taking last part of current chunk
                    words = current_chunk.split()
                    overlap_words = []
                    overlap_tokens = 0
                    
                    # Work backwards to find overlap
                    for word in reversed(words):
                        test_overlap = " ".join([word] + overlap_words)
                        if self.count_tokens(test_overlap) <= self.overlap:
                            overlap_words.insert(0, word)
                            overlap_tokens = self.count_tokens(test_overlap)
                        else:
                            break
                    
                    current_chunk = " ".join(overlap_words) + "\n\n" + paragraph if overlap_words else paragraph
                else:
                    current_chunk = paragraph
        
        # Handle final chunk
        if current_chunk and self.count_tokens(current_chunk) >= self.min_tokens:
            chunks.append({
                "chunk_id": chunk_id,
                "novel_name": novel_name,
                "text": current_chunk,
                "token_count": self.count_tokens(current_chunk)
            })
        
        return chunks
    
    def process_novels_directory(self, novels_dir: str) -> List[Dict[str, Any]]:
        """
        Process all novels in a directory.
        
        Args:
            novels_dir: Directory containing novel .txt files
            
        Returns:
            List of all chunks from all novels
        """
        all_chunks = []
        
        for filename in os.listdir(novels_dir):
            if filename.endswith('.txt'):
                novel_path = os.path.join(novels_dir, filename)
                novel_name = filename.replace('.txt', '')
                
                print(f"Processing novel: {novel_name}")
                text = self.load_novel(novel_path)
                chunks = self.chunk_text(text, novel_name)
                all_chunks.extend(chunks)
                
                print(f"Generated {len(chunks)} chunks from {novel_name}")
        
        return all_chunks
    
    def create_pathway_table(self, chunks: List[Dict[str, Any]]) -> pw.Table:
        """
        Create a Pathway table from chunks.
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            Pathway table with chunk data
        """
        schema = {
            "chunk_id": pw.column_definition(int),
            "novel_name": pw.column_definition(str),
            "text": pw.column_definition(str),
            "token_count": pw.column_definition(int)
        }
        
        return pw.Table.from_records(chunks, schema=schema)
