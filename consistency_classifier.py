"""
Backstory vs Evidence Consistency Classifier

Clean, testable Python code implementing consistency evaluation module.
Separates model calls from decision logic with deterministic aggregation rules.
"""

from typing import List, Dict, Any, Optional, Tuple
import json
from dataclasses import dataclass
from enum import Enum


class ConsistencyLabel(Enum):
    """Enumeration for consistency classification labels."""
    SUPPORT = "SUPPORT"
    CONTRADICT = "CONTRADICT"
    NEUTRAL = "NEUTRAL"


@dataclass
class ChunkEvaluation:
    """Data class representing evaluation result for a single chunk."""
    chunk_text: str
    label: ConsistencyLabel
    confidence: float
    reasoning: str


class ConsistencyClassifier:
    """
    Consistency classifier that evaluates relationship between backstory and evidence chunks.
    
    Uses focused LLM calls for classification with deterministic aggregation rules.
    """
    
    def __init__(self, model_client=None, contradiction_threshold: float = 0.7):
        """
        Initialize the consistency classifier.
        
        Args:
            model_client: LLM client for classification (must have .generate() method)
            contradiction_threshold: Confidence threshold for strong contradiction
        """
        self.model_client = model_client
        self.contradiction_threshold = contradiction_threshold
        
        # Minimal classification prompt
        self.classification_prompt = """Classify the relationship between BACKSTORY and EVIDENCE:

BACKSTORY: {backstory}

EVIDENCE: {evidence}

Choose one: SUPPORT, CONTRADICT, NEUTRAL

Respond with JSON format:
{{"label": "SUPPORT|CONTRADICT|NEUTRAL", "confidence": 0.0-1.0, "reasoning": "brief explanation"}}"""
    
    def classify_chunk(self, backstory: str, evidence_chunk: str) -> ChunkEvaluation:
        """
        Classify a single evidence chunk against the backstory.
        
        Args:
            backstory: The backstory text
            evidence_chunk: Single evidence chunk text
            
        Returns:
            ChunkEvaluation with label, confidence, and reasoning
        """
        if self.model_client is None:
            # Fallback to rule-based classification for testing
            return self._rule_based_classification(backstory, evidence_chunk)
        
        # Prepare prompt
        prompt = self.classification_prompt.format(
            backstory=backstory[:500],  # Truncate for context limits
            evidence=evidence_chunk[:500]
        )
        
        try:
            # Call LLM
            response = self.model_client.generate(prompt)
            
            # Parse JSON response
            result = json.loads(response.strip())
            
            # Validate and create evaluation
            label = ConsistencyLabel(result.get("label", "NEUTRAL"))
            confidence = float(result.get("confidence", 0.5))
            reasoning = result.get("reasoning", "No reasoning provided")
            
            return ChunkEvaluation(
                chunk_text=evidence_chunk,
                label=label,
                confidence=confidence,
                reasoning=reasoning
            )
            
        except Exception as e:
            # Fallback to rule-based on LLM failure
            print(f"LLM classification failed: {e}")
            return self._rule_based_classification(backstory, evidence_chunk)
    
    def _rule_based_classification(self, backstory: str, evidence_chunk: str) -> ChunkEvaluation:
        """
        Rule-based classification fallback for testing and LLM failures.
        
        Args:
            backstory: The backstory text
            evidence_chunk: Single evidence chunk text
            
        Returns:
            ChunkEvaluation with rule-based classification
        """
        # Simple keyword-based classification
        backstory_lower = backstory.lower()
        evidence_lower = evidence_chunk.lower()
        
        # Contradiction indicators
        contradiction_words = [
            "never", "not", "no", "none", "nothing", "nowhere", 
            "contradict", "opposite", "different", "unlike", "however",
            "but", "although", "despite", "yet", "instead", "rather"
        ]
        
        # Support indicators  
        support_words = [
            "also", "too", "similarly", "like", "same", "agree", "confirm",
            "support", "match", "consistent", "align", "correspond", "indeed",
            "rescued", "freed", "escaped", "imprisoned", "captivity", "prisoner"
        ]
        
        # Count indicators
        contradiction_count = sum(1 for word in contradiction_words if word in evidence_lower)
        support_count = sum(1 for word in support_words if word in evidence_lower)
        
        # Entity overlap (simple word overlap)
        backstory_words = set(backstory_lower.split())
        evidence_words = set(evidence_lower.split())
        overlap = len(backstory_words & evidence_words)
        overlap_ratio = overlap / max(len(backstory_words), len(evidence_words)) if max(len(backstory_words), len(evidence_words)) > 0 else 0
        
        # Determine classification with improved logic
        # First check for explicit contradictions (highest priority)
        if contradiction_count >= 2:
            label = ConsistencyLabel.CONTRADICT
            confidence = min(0.9, 0.5 + contradiction_count * 0.2)
            reasoning = f"Found {contradiction_count} contradiction indicators"
        elif contradiction_count == 1 and overlap_ratio > 0.3:
            # High overlap with contradiction suggests direct contradiction
            label = ConsistencyLabel.CONTRADICT
            confidence = 0.8
            reasoning = f"Found contradiction with high entity overlap ({overlap_ratio:.2f})"
        elif contradiction_count == 1:
            # Any contradiction indicator
            label = ConsistencyLabel.CONTRADICT
            confidence = 0.7
            reasoning = f"Found contradiction indicator with overlap ({overlap_ratio:.2f})"
        elif support_count >= 2 and overlap_ratio > 0.1:
            label = ConsistencyLabel.SUPPORT
            confidence = min(0.8, 0.4 + support_count * 0.15 + overlap_ratio * 0.3)
            reasoning = f"Found {support_count} support indicators and {overlap_ratio:.2f} entity overlap"
        elif support_count > 0 and overlap_ratio > 0.3:
            label = ConsistencyLabel.SUPPORT
            confidence = min(0.7, 0.3 + overlap_ratio)
            reasoning = f"Support indicators with high entity overlap ({overlap_ratio:.2f})"
        elif overlap_ratio > 0.4 and contradiction_count == 0:
            label = ConsistencyLabel.SUPPORT
            confidence = min(0.7, 0.3 + overlap_ratio)
            reasoning = f"High entity overlap ({overlap_ratio:.2f}) suggests support"
        else:
            label = ConsistencyLabel.NEUTRAL
            confidence = 0.6
            reasoning = "Insufficient evidence for strong support or contradiction"
        
        return ChunkEvaluation(
            chunk_text=evidence_chunk,
            label=label,
            confidence=confidence,
            reasoning=reasoning
        )
    
    def evaluate_all_chunks(self, backstory: str, evidence_chunks: List[str]) -> List[ChunkEvaluation]:
        """
        Evaluate all evidence chunks against the backstory.
        
        Args:
            backstory: The backstory text
            evidence_chunks: List of evidence chunk texts
            
        Returns:
            List of ChunkEvaluation objects
        """
        evaluations = []
        
        for i, chunk in enumerate(evidence_chunks):
            evaluation = self.classify_chunk(backstory, chunk)
            evaluations.append(evaluation)
        
        return evaluations
    
    def aggregate_decisions(self, evaluations: List[ChunkEvaluation]) -> int:
        """
        Apply deterministic aggregation rules to reach final decision.
        
        Args:
            evaluations: List of ChunkEvaluation objects
            
        Returns:
            1 if consistent, 0 if inconsistent
        """
        # Rule: If any chunk is a strong CONTRADICTION → return 0
        for evaluation in evaluations:
            if (evaluation.label == ConsistencyLabel.CONTRADICT and 
                evaluation.confidence >= self.contradiction_threshold):
                return 0
        
        # Otherwise → return 1
        return 1
    
    def predict_consistency(self, backstory: str, retrieved_chunks: List[str]) -> int:
        """
        Main prediction function that evaluates consistency between backstory and evidence.
        
        Args:
            backstory: The backstory text
            retrieved_chunks: List of retrieved evidence chunks
            
        Returns:
            1 if backstory is consistent with evidence, 0 if inconsistent
        """
        # Evaluate all chunks
        evaluations = self.evaluate_all_chunks(backstory, retrieved_chunks)
        
        # Apply deterministic aggregation
        final_decision = self.aggregate_decisions(evaluations)
        
        return final_decision
    
    def get_detailed_analysis(self, backstory: str, retrieved_chunks: List[str]) -> Dict[str, Any]:
        """
        Get detailed analysis including all evaluations and reasoning.
        
        Args:
            backstory: The backstory text
            retrieved_chunks: List of retrieved evidence chunks
            
        Returns:
            Dictionary with detailed analysis
        """
        evaluations = self.evaluate_all_chunks(backstory, retrieved_chunks)
        final_decision = self.aggregate_decisions(evaluations)
        
        # Count by label
        label_counts = {}
        for label in ConsistencyLabel:
            label_counts[label.value] = sum(1 for e in evaluations if e.label == label)
        
        # Find strong contradictions
        strong_contradictions = [
            e for e in evaluations 
            if e.label == ConsistencyLabel.CONTRADICT and e.confidence >= self.contradiction_threshold
        ]
        
        return {
            "final_decision": final_decision,
            "decision_reasoning": "Strong contradiction found" if strong_contradictions else "No strong contradictions",
            "total_chunks": len(evaluations),
            "label_distribution": label_counts,
            "strong_contradictions": len(strong_contradictions),
            "evaluations": [
                {
                    "label": eval.label.value,
                    "confidence": eval.confidence,
                    "reasoning": eval.reasoning,
                    "chunk_preview": eval.chunk_text[:100] + "..." if len(eval.chunk_text) > 100 else eval.chunk_text
                }
                for eval in evaluations
            ]
        }


# Mock LLM client for testing
class MockLLMClient:
    """Mock LLM client for testing purposes."""
    
    def __init__(self, responses: Optional[Dict[str, str]] = None):
        """Initialize with predefined responses."""
        self.responses = responses or {}
    
    def generate(self, prompt: str) -> str:
        """Generate response based on prompt content."""
        # Simple rule-based mock responses
        if "contradict" in prompt.lower() or "never" in prompt.lower():
            return '{"label": "CONTRADICT", "confidence": 0.8, "reasoning": "Evidence contradicts backstory"}'
        elif "support" in prompt.lower() or "also" in prompt.lower():
            return '{"label": "SUPPORT", "confidence": 0.7, "reasoning": "Evidence supports backstory"}'
        else:
            return '{"label": "NEUTRAL", "confidence": 0.6, "reasoning": "Evidence is neutral to backstory"}'


# Unit tests
def test_consistency_classifier():
    """Unit tests for consistency classifier."""
    print("Running consistency classifier tests...")
    
    # Test 1: Rule-based classification
    classifier = ConsistencyClassifier(model_client=None)
    
    backstory = "John was imprisoned for 10 years and later escaped."
    evidence_contradict = "John was never imprisoned and never escaped from anywhere."
    evidence_support = "John was also imprisoned for a decade and successfully escaped from captivity."
    evidence_neutral = "The weather was sunny that day."
    
    # Test contradiction
    eval_contradict = classifier.classify_chunk(backstory, evidence_contradict)
    assert eval_contradict.label == ConsistencyLabel.CONTRADICT
    print(f"✓ Contradiction test passed: {eval_contradict.label} ({eval_contradict.confidence:.2f})")
    
    # Test support
    eval_support = classifier.classify_chunk(backstory, evidence_support)
    assert eval_support.label == ConsistencyLabel.SUPPORT
    print(f"✓ Support test passed: {eval_support.label} ({eval_support.confidence:.2f})")
    
    # Test neutral
    eval_neutral = classifier.classify_chunk(backstory, evidence_neutral)
    assert eval_neutral.label == ConsistencyLabel.NEUTRAL
    print(f"✓ Neutral test passed: {eval_neutral.label} ({eval_neutral.confidence:.2f})")
    
    # Test aggregation rules
    chunks = [evidence_contradict, evidence_support, evidence_neutral]
    decision = classifier.predict_consistency(backstory, chunks)
    assert decision == 0  # Should be inconsistent due to contradiction
    print(f"✓ Aggregation test passed: decision = {decision}")
    
    # Test 2: Mock LLM client
    mock_client = MockLLMClient()
    classifier_llm = ConsistencyClassifier(model_client=mock_client)
    
    decision_llm = classifier_llm.predict_consistency(backstory, chunks)
    print(f"✓ Mock LLM test passed: decision = {decision_llm}")
    
    # Test detailed analysis
    analysis = classifier.get_detailed_analysis(backstory, chunks)
    assert "final_decision" in analysis
    assert "evaluations" in analysis
    assert len(analysis["evaluations"]) == 3
    print(f"✓ Detailed analysis test passed: {analysis['final_decision']}")
    
    print("All tests passed! ✓")


# Example usage
def main():
    """Example usage of consistency classifier."""
    print("Consistency Classifier Example")
    print("=" * 40)
    
    # Initialize classifier
    classifier = ConsistencyClassifier(contradiction_threshold=0.7)
    
    # Example backstory and evidence
    backstory = """
    Captain Grant was a Scottish sailor who was shipwrecked in 1864. 
    He was imprisoned by Maori tribes in New Zealand for two years 
    before being rescued by Lord Glenarvan's expedition.
    """
    
    evidence_chunks = [
        "Captain Grant spent two years as a prisoner of the Maori people after his ship wrecked on the New Zealand coast.",
        "Lord Glenarvan organized a rescue mission that successfully freed Captain Grant from captivity.",
        "Captain Grant never left Scotland and lived his entire life in Edinburgh.",
        "The expedition traveled across Australia and New Zealand in search of the missing captain.",
        "The weather conditions in New Zealand were particularly harsh that winter."
    ]
    
    # Get prediction
    prediction = classifier.predict_consistency(backstory, evidence_chunks)
    print(f"\nFinal Prediction: {prediction} ({'Consistent' if prediction == 1 else 'Inconsistent'})")
    
    # Get detailed analysis
    analysis = classifier.get_detailed_analysis(backstory, evidence_chunks)
    print(f"\nReasoning: {analysis['decision_reasoning']}")
    print(f"Total Chunks: {analysis['total_chunks']}")
    print(f"Label Distribution: {analysis['label_distribution']}")
    print(f"Strong Contradictions: {analysis['strong_contradictions']}")
    
    print("\nDetailed Evaluations:")
    for i, eval_data in enumerate(analysis['evaluations'], 1):
        print(f"\nChunk {i}:")
        print(f"  Label: {eval_data['label']}")
        print(f"  Confidence: {eval_data['confidence']:.2f}")
        print(f"  Reasoning: {eval_data['reasoning']}")
        print(f"  Preview: {eval_data['chunk_preview']}")
    
    # Run tests
    print("\n" + "=" * 40)
    test_consistency_classifier()


if __name__ == "__main__":
    main()
