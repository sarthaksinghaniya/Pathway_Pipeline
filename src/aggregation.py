"""
Deterministic aggregation rules for binary classification based on chunk evaluations.
"""

from typing import List, Dict, Any, Tuple
from enum import Enum


class ClassificationDecision(Enum):
    """Binary classification decisions."""
    CONSISTENT = 1
    INCONSISTENT = 0


class ConsistencyAggregator:
    """Applies deterministic rules to aggregate chunk evaluations into binary classification."""
    
    def __init__(self, contradiction_threshold: float = 0.4, support_threshold: float = 0.3):
        """
        Initialize the aggregator with decision thresholds.
        
        Args:
            contradiction_threshold: Minimum contradiction confidence to trigger inconsistency
            support_threshold: Minimum support confidence to consider as supporting evidence
        """
        self.contradiction_threshold = contradiction_threshold
        self.support_threshold = support_threshold
    
    def has_strong_contradiction(self, evaluations: List[Dict[str, Any]]) -> bool:
        """
        Check if there's any strong contradiction evidence.
        
        Args:
            evaluations: List of chunk evaluation results
            
        Returns:
            True if strong contradiction found
        """
        for evaluation in evaluations:
            if (evaluation["relationship"] == "CONTRADICT" and 
                evaluation["confidence"] >= self.contradiction_threshold):
                return True
        return False
    
    def count_supporting_evidence(self, evaluations: List[Dict[str, Any]]) -> int:
        """
        Count chunks that support the backstory.
        
        Args:
            evaluations: List of chunk evaluation results
            
        Returns:
            Number of supporting chunks
        """
        return sum(1 for eval_result in evaluations 
                  if eval_result["relationship"] == "SUPPORT" and 
                  eval_result["confidence"] >= self.support_threshold)
    
    def count_contradicting_evidence(self, evaluations: List[Dict[str, Any]]) -> int:
        """
        Count chunks that contradict the backstory.
        
        Args:
            evaluations: List of chunk evaluation results
            
        Returns:
            Number of contradicting chunks
        """
        return sum(1 for eval_result in evaluations 
                  if eval_result["relationship"] == "CONTRADICT" and 
                  eval_result["confidence"] >= self.contradiction_threshold)
    
    def calculate_evidence_balance(self, evaluations: List[Dict[str, Any]]) -> Tuple[float, float]:
        """
        Calculate the balance between supporting and contradicting evidence.
        
        Args:
            evaluations: List of chunk evaluation results
            
        Returns:
            Tuple of (support_strength, contradiction_strength)
        """
        support_strength = 0.0
        contradiction_strength = 0.0
        
        for evaluation in evaluations:
            if evaluation["relationship"] == "SUPPORT":
                support_strength += evaluation["confidence"]
            elif evaluation["relationship"] == "CONTRADICT":
                contradiction_strength += evaluation["confidence"]
        
        return support_strength, contradiction_strength
    
    def apply_deterministic_rules(self, evaluations: List[Dict[str, Any]]) -> ClassificationDecision:
        """
        Apply deterministic aggregation rules for binary classification.
        
        Rules:
        1. Any strong contradiction → INCONSISTENT (0)
        2. Otherwise → CONSISTENT (1)
        
        Args:
            evaluations: List of chunk evaluation results
            
        Returns:
            Classification decision
        """
        if not evaluations:
            # No evidence found, default to consistent
            return ClassificationDecision.CONSISTENT
        
        # Rule 1: Check for strong contradictions
        if self.has_strong_contradiction(evaluations):
            return ClassificationDecision.INCONSISTENT
        
        # Rule 2: No strong contradictions found
        return ClassificationDecision.CONSISTENT
    
    def apply_enhanced_rules(self, evaluations: List[Dict[str, Any]]) -> Tuple[ClassificationDecision, Dict[str, Any]]:
        """
        Apply enhanced deterministic rules with detailed reasoning.
        
        Args:
            evaluations: List of chunk evaluation results
            
        Returns:
            Tuple of (classification decision, reasoning details)
        """
        if not evaluations:
            reasoning = {
                "decision": "CONSISTENT",
                "reasoning": "No evidence found - defaulting to consistent",
                "support_count": 0,
                "contradiction_count": 0,
                "neutral_count": 0,
                "strong_contradiction": False
            }
            return ClassificationDecision.CONSISTENT, reasoning
        
        # Count evidence types
        support_count = self.count_supporting_evidence(evaluations)
        contradiction_count = self.count_contradicting_evidence(evaluations)
        neutral_count = len(evaluations) - support_count - contradiction_count
        
        # Calculate evidence balance
        support_strength, contradiction_strength = self.calculate_evidence_balance(evaluations)
        
        # Check for strong contradictions
        has_strong_contradiction = self.has_strong_contradiction(evaluations)
        
        # Apply primary rule
        if has_strong_contradiction:
            decision = ClassificationDecision.INCONSISTENT
            primary_reason = f"Strong contradiction found (threshold: {self.contradiction_threshold})"
        else:
            decision = ClassificationDecision.CONSISTENT
            primary_reason = "No strong contradictions detected"
        
        reasoning = {
            "decision": decision.name,
            "primary_reason": primary_reason,
            "support_count": support_count,
            "contradiction_count": contradiction_count,
            "neutral_count": neutral_count,
            "support_strength": support_strength,
            "contradiction_strength": contradiction_strength,
            "strong_contradiction": has_strong_contradiction,
            "total_chunks": len(evaluations)
        }
        
        return decision, reasoning
    
    def batch_classify(self, batch_evaluations: List[List[Dict[str, Any]]]) -> List[Tuple[ClassificationDecision, Dict[str, Any]]]:
        """
        Classify multiple backstories based on their chunk evaluations.
        
        Args:
            batch_evaluations: List of evaluation lists for each backstory
            
        Returns:
            List of (decision, reasoning) tuples
        """
        results = []
        
        for evaluations in batch_evaluations:
            decision, reasoning = self.apply_enhanced_rules(evaluations)
            results.append((decision, reasoning))
        
        return results
    
    def get_classification_stats(self, results: List[Tuple[ClassificationDecision, Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Get statistics about classification results.
        
        Args:
            results: List of classification results
            
        Returns:
            Statistics dictionary
        """
        if not results:
            return {"total": 0, "consistent": 0, "inconsistent": 0, "consistent_rate": 0.0}
        
        decisions = [decision for decision, _ in results]
        
        consistent_count = sum(1 for d in decisions if d == ClassificationDecision.CONSISTENT)
        inconsistent_count = sum(1 for d in decisions if d == ClassificationDecision.INCONSISTENT)
        
        stats = {
            "total": len(results),
            "consistent": consistent_count,
            "inconsistent": inconsistent_count,
            "consistent_rate": consistent_count / len(results),
            "inconsistent_rate": inconsistent_count / len(results)
        }
        
        # Calculate average evidence counts
        all_reasonings = [reasoning for _, reasoning in results]
        stats["avg_support_count"] = sum(r["support_count"] for r in all_reasonings) / len(all_reasonings)
        stats["avg_contradiction_count"] = sum(r["contradiction_count"] for r in all_reasonings) / len(all_reasonings)
        stats["avg_neutral_count"] = sum(r["neutral_count"] for r in all_reasonings) / len(all_reasonings)
        
        return stats
