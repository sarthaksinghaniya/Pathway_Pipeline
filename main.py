"""
End-to-end pipeline for novel-backstory consistency classification.
Kharagpur Data Science Hackathon - Track A

This script processes the dataset, runs the complete pipeline, and generates results.csv.
"""

import os
import sys
import pandas as pd
import argparse
from pathlib import Path

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.ingestion import NovelChunker
from src.vector_store import SemanticVectorStore
from src.retrieval import BackstoryRetriever
from src.reasoning import ConsistencyReasoner
from src.aggregation import ConsistencyAggregator


class ConsistencyPipeline:
    """End-to-end pipeline for backstory-novel consistency classification."""
    
    def __init__(self, novels_dir: str, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the pipeline.
        
        Args:
            novels_dir: Directory containing novel .txt files
            model_name: Name of the sentence transformer model
        """
        self.novels_dir = novels_dir
        self.model_name = model_name
        
        # Initialize components
        self.chunker = NovelChunker()
        self.vector_store = SemanticVectorStore(model_name)
        self.retriever = BackstoryRetriever(self.vector_store)
        self.reasoner = ConsistencyReasoner(model_name)
        self.aggregator = ConsistencyAggregator()
        
        # Processed data
        self.chunks_table = None
        self.embeddings_table = None
        self.indexed_table = None
        
    def setup_vector_store(self):
        """Set up the vector store with novel chunks."""
        print("Setting up vector store...")
        
        # Load and chunk novels
        print("Loading and chunking novels...")
        chunks = self.chunker.process_novels_directory(self.novels_dir)
        print(f"Generated {len(chunks)} total chunks")
        
        # Create Pathway table
        self.chunks_table = self.chunker.create_pathway_table(chunks)
        
        # Create embeddings
        print("Creating embeddings...")
        self.embeddings_table = self.vector_store.create_embeddings_table(self.chunks_table)
        
        # Set up vector index
        self.indexed_table = self.vector_store.setup_vector_index(self.embeddings_table)
        
        # Set up retriever
        self.retriever.set_indexed_table(self.indexed_table)
        
        print("Vector store setup complete!")
    
    def process_backstories(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process backstories and generate predictions.
        
        Args:
            df: DataFrame with backstories
            
        Returns:
            DataFrame with added predictions and reasoning
        """
        print(f"Processing {len(df)} backstories...")
        
        # Extract backstories and novel names
        backstories = df['content'].tolist()
        novel_names = df['book_name'].tolist() if 'book_name' in df.columns else None
        
        # Step 1: Retrieve relevant chunks for each backstory
        print("Retrieving relevant chunks...")
        retrieval_results = self.retriever.retrieve_for_backstories_batch(backstories, novel_names)
        
        # Step 2: Evaluate consistency for each backstory
        print("Evaluating consistency...")
        all_evaluations = []
        
        for i, (backstory, retrieved_chunks) in enumerate(zip(backstories, retrieval_results)):
            if i % 10 == 0:
                print(f"Processing backstory {i+1}/{len(backstories)}")
            
            evaluations = self.reasoner.evaluate_retrieved_chunks(backstory, retrieved_chunks)
            all_evaluations.append(evaluations)
        
        # Step 3: Apply aggregation rules for binary classification
        print("Applying aggregation rules...")
        classification_results = self.aggregator.batch_classify(all_evaluations)
        
        # Add results to DataFrame
        df_copy = df.copy()
        df_copy['prediction'] = [decision.value for decision, _ in classification_results]
        df_copy['reasoning'] = [reasoning for _, reasoning in classification_results]
        df_copy['evaluations'] = all_evaluations
        df_copy['retrieved_chunks'] = retrieval_results
        
        return df_copy
    
    def run_pipeline(self, input_csv: str, output_csv: str) -> pd.DataFrame:
        """
        Run the complete end-to-end pipeline.
        
        Args:
            input_csv: Path to input CSV file
            output_csv: Path to output CSV file
            
        Returns:
            DataFrame with results
        """
        print("Starting consistency classification pipeline...")
        
        # Setup vector store
        self.setup_vector_store()
        
        # Load input data
        print(f"Loading data from {input_csv}...")
        df = pd.read_csv(input_csv)
        print(f"Loaded {len(df)} rows")
        
        # Process backstories
        results_df = self.process_backstories(df)
        
        # Save results
        print(f"Saving results to {output_csv}...")
        
        # Create output DataFrame with required columns
        output_df = results_df[['id', 'prediction']].copy()
        output_df.to_csv(output_csv, index=False)
        
        # Also save detailed results
        detailed_output = output_csv.replace('.csv', '_detailed.csv')
        results_df.to_csv(detailed_output, index=False)
        
        print(f"Results saved to {output_csv}")
        print(f"Detailed results saved to {detailed_output}")
        
        # Print statistics
        self.print_statistics(results_df)
        
        return results_df
    
    def print_statistics(self, results_df: pd.DataFrame):
        """Print pipeline statistics."""
        print("\n" + "="*50)
        print("PIPELINE STATISTICS")
        print("="*50)
        
        # Classification statistics
        predictions = results_df['prediction'].value_counts()
        print(f"Total predictions: {len(results_df)}")
        print(f"Consistent (1): {predictions.get(1, 0)}")
        print(f"Inconsistent (0): {predictions.get(0, 0)}")
        print(f"Consistent rate: {predictions.get(1, 0) / len(results_df):.3f}")
        
        # Retrieval statistics
        all_retrievals = results_df['retrieved_chunks'].tolist()
        retrieval_stats = self.retriever.get_retrieval_stats(all_retrievals)
        print(f"\nRetrieval Statistics:")
        print(f"Average chunks per query: {retrieval_stats.get('avg_chunks', 0):.1f}")
        print(f"Max chunks: {retrieval_stats.get('max_chunks', 0)}")
        print(f"Min chunks: {retrieval_stats.get('min_chunks', 0)}")
        if 'avg_similarity' in retrieval_stats:
            print(f"Average similarity: {retrieval_stats['avg_similarity']:.3f}")
        
        # Classification reasoning statistics
        all_reasonings = results_df['reasoning'].tolist()
        classification_stats = self.aggregator.get_classification_stats(
            [(r['decision'], r) for r in all_reasonings]
        )
        print(f"\nClassification Statistics:")
        print(f"Average support count: {classification_stats.get('avg_support_count', 0):.1f}")
        print(f"Average contradiction count: {classification_stats.get('avg_contradiction_count', 0):.1f}")
        print(f"Average neutral count: {classification_stats.get('avg_neutral_count', 0):.1f}")


def main():
    """Main function to run the pipeline."""
    parser = argparse.ArgumentParser(description='Novel-Backstory Consistency Classification Pipeline')
    parser.add_argument('--input', type=str, default='data/test.csv', 
                       help='Input CSV file path')
    parser.add_argument('--output', type=str, default='results.csv',
                       help='Output CSV file path')
    parser.add_argument('--novels-dir', type=str, default='data/novel',
                       help='Directory containing novel .txt files')
    parser.add_argument('--model', type=str, default='all-MiniLM-L6-v2',
                       help='Sentence transformer model name')
    
    args = parser.parse_args()
    
    # Validate input paths
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} does not exist")
        return 1
    
    if not os.path.exists(args.novels_dir):
        print(f"Error: Novels directory {args.novels_dir} does not exist")
        return 1
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Initialize and run pipeline
    pipeline = ConsistencyPipeline(args.novels_dir, args.model)
    
    try:
        results = pipeline.run_pipeline(args.input, args.output)
        print("\nPipeline completed successfully!")
        return 0
    except Exception as e:
        print(f"Error running pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
