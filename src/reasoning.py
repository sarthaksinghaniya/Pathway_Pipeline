"""
Reasoning module to evaluate SUPPORT/CONTRADICT/NEUTRAL relationships between backstories and novel chunks.
"""

import re
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
import numpy as np


class ConsistencyReasoner:
    """Evaluates whether novel chunks support, contradict, or are neutral to backstories."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the reasoner.
        
        Args:
            model_name: Name of the sentence transformer model
        """
        self.model = SentenceTransformer(model_name)
        
        # Keywords for contradiction detection
        self.contradiction_keywords = [
            "never", "not", "no", "none", "nothing", "nowhere", "neither", "nor",
            "contradict", "opposite", "different", "unlike", "however", "but",
            "although", "despite", "yet", "instead", "rather", "whereas", "while"
        ]
        
        # Keywords for support detection  
        self.support_keywords = [
            "also", "too", "similarly", "like", "same", "agree", "confirm",
            "support", "match", "consistent", "align", "correspond", "indeed"
        ]
    
    def extract_entities_and_events(self, text: str) -> List[str]:
        """
        Extract key entities and events from text.
        
        Args:
            text: Input text
            
        Returns:
            List of extracted entities/events
        """
        # Simple extraction of capitalized words and numbers
        entities = []
        
        # Find capitalized words (potential names/places)
        capitalized = re.findall(r'\b[A-Z][a-z]+\b', text)
        entities.extend(capitalized)
        
        # Find years and numbers
        numbers = re.findall(r'\b\d{4}\b|\b\d+\b', text)
        entities.extend(numbers)
        
        # Find action verbs (simplified)
        actions = re.findall(r'\b(killed|died|born|married|escaped|imprisoned|arrested|fought|traveled|lived|worked)\b', text, re.IGNORECASE)
        entities.extend([action.lower() for action in actions])
        
        return list(set(entities))  # Remove duplicates
    
    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        emb1 = self.model.encode(text1)
        emb2 = self.model.encode(text2)
        
        # Cosine similarity
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return float(similarity)
    
    def detect_contradiction_signals(self, backstory: str, chunk: str) -> float:
        """
        Detect linguistic signals of contradiction.
        
        Args:
            backstory: Backstory text
            chunk: Novel chunk text
            
        Returns:
            Contradiction signal score (0-1)
        """
        contradiction_score = 0.0
        
        # Check for contradiction keywords in chunk
        chunk_lower = chunk.lower()
        for keyword in self.contradiction_keywords:
            if keyword in chunk_lower:
                contradiction_score += 0.1
        
        # Check for negation patterns
        negation_patterns = [r'\bnot\b', r'\bnever\b', r'\bno\b', r'\bnone\b']
        for pattern in negation_patterns:
            matches = len(re.findall(pattern, chunk_lower))
            contradiction_score += matches * 0.05
        
        # Check for direct opposition (simplified)
        backstory_entities = self.extract_entities_and_events(backstory)
        chunk_entities = self.extract_entities_and_events(chunk)
        
        # If same entities appear with contradictory actions
        common_entities = set(backstory_entities) & set(chunk_entities)
        if common_entities:
            contradiction_score += 0.2
        
        return min(contradiction_score, 1.0)
    
    def detect_support_signals(self, backstory: str, chunk: str) -> float:
        """
        Detect linguistic signals of support.
        
        Args:
            backstory: Backstory text
            chunk: Novel chunk text
            
        Returns:
            Support signal score (0-1)
        """
        support_score = 0.0
        
        # Check for support keywords
        chunk_lower = chunk.lower()
        for keyword in self.support_keywords:
            if keyword in chunk_lower:
                support_score += 0.1
        
        # Check for entity overlap
        backstory_entities = self.extract_entities_and_events(backstory)
        chunk_entities = self.extract_entities_and_events(chunk)
        
        common_entities = set(backstory_entities) & set(chunk_entities)
        if common_entities:
            overlap_ratio = len(common_entities) / max(len(backstory_entities), len(chunk_entities))
            support_score += overlap_ratio * 0.5
        
        return min(support_score, 1.0)
    
    def evaluate_chunk_consistency(self, backstory: str, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate whether a chunk supports, contradicts, or is neutral to a backstory.
        
        Args:
            backstory: The backstory text
            chunk: Chunk dictionary with text and metadata
            
        Returns:
            Dictionary with evaluation results
        """
        chunk_text = chunk["text"]
        
        # Calculate semantic similarity
        semantic_sim = self.calculate_semantic_similarity(backstory, chunk_text)
        
        # Detect contradiction and support signals
        contradiction_score = self.detect_contradiction_signals(backstory, chunk_text)
        support_score = self.detect_support_signals(backstory, chunk_text)
        
        # Determine relationship
        if contradiction_score > 0.3 and contradiction_score > support_score:
            relationship = "CONTRADICT"
            confidence = contradiction_score
        elif support_score > 0.3 and support_score > contradiction_score:
            relationship = "SUPPORT"
            confidence = support_score
        else:
            relationship = "NEUTRAL"
            confidence = 1.0 - max(contradiction_score, support_score)
        
        return {
            "chunk_id": chunk["chunk_id"],
            "novel_name": chunk["novel_name"],
            "relationship": relationship,
            "confidence": confidence,
            "semantic_similarity": semantic_sim,
            "contradiction_score": contradiction_score,
            "support_score": support_score,
            "chunk_text": chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text
        }
    
    def evaluate_retrieved_chunks(self, backstory: str, retrieved_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Evaluate consistency for all retrieved chunks.
        
        Args:
            backstory: The backstory text
            retrieved_chunks: List of retrieved chunk dictionaries
            
        Returns:
            List of evaluation results for each chunk
        """
        evaluations = []
        
        for chunk in retrieved_chunks:
            evaluation = self.evaluate_chunk_consistency(backstory, chunk)
            evaluations.append(evaluation)
        
        return evaluations
    
    def get_reasoning_summary(self, evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get summary statistics for chunk evaluations.
        
        Args:
            evaluations: List of chunk evaluation results
            
        Returns:
            Summary dictionary
        """
        if not evaluations:
            return {"total_chunks": 0, "support_count": 0, "contradict_count": 0, "neutral_count": 0}
        
        relationships = [eval_result["relationship"] for eval_result in evaluations]
        
        summary = {
            "total_chunks": len(evaluations),
            "support_count": relationships.count("SUPPORT"),
            "contradict_count": relationships.count("CONTRADICT"),
            "neutral_count": relationships.count("NEUTRAL"),
            "avg_confidence": sum(e["confidence"] for e in evaluations) / len(evaluations),
            "avg_semantic_similarity": sum(e["semantic_similarity"] for e in evaluations) / len(evaluations)
        }
        
        # Add confidence scores by relationship type
        for rel_type in ["SUPPORT", "CONTRADICT", "NEUTRAL"]:
            rel_evals = [e for e in evaluations if e["relationship"] == rel_type]
            if rel_evals:
                summary[f"avg_{rel_type.lower()}_confidence"] = sum(e["confidence"] for e in rel_evals) / len(rel_evals)
            else:
                summary[f"avg_{rel_type.lower()}_confidence"] = 0.0
        
        return summary
