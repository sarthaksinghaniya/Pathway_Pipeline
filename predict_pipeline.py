#!/usr/bin/env python3
"""
End-to-End Prediction Pipeline

Clean, reproducible Python script that performs end-to-end inference.
Integrates Pathway retrieval and consistency classification modules.

Usage:
    python predict_pipeline.py --input data/test.csv --output results/results.csv --k 5
"""

import argparse
import logging
import os
import sys
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict, Any
import time
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import modules
try:
    from pathway_final import retrieve_chunks
    from consistency_final import predict_consistency
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please ensure pathway_final.py and consistency_final.py are in the same directory.")
    sys.exit(1)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('pipeline.log')
    ]
)
logger = logging.getLogger(__name__)


class PredictionPipeline:
    """End-to-end prediction pipeline for backstory consistency evaluation."""
    
    def __init__(self, k: int = 5, batch_size: Optional[int] = None):
        """
        Initialize the prediction pipeline.
        
        Args:
            k: Number of chunks to retrieve for each backstory (default: 5)
            batch_size: Optional batch size for processing (None for streaming)
        """
        self.k = k
        self.batch_size = batch_size
        self.results = []
        
        # Statistics tracking
        self.stats = {
            'total_processed': 0,
            'successful_predictions': 0,
            'failed_predictions': 0,
            'empty_backstories': 0,
            'empty_retrievals': 0,
            'start_time': None,
            'end_time': None
        }
        
        logger.info(f"Initialized prediction pipeline with k={k}")
        if batch_size:
            logger.info(f"Using batch size: {batch_size}")
    
    def load_test_data(self, input_path: str) -> pd.DataFrame:
        """
        Load test data from CSV file.
        
        Args:
            input_path: Path to test.csv file
            
        Returns:
            DataFrame with test data
            
        Raises:
            FileNotFoundError: If input file doesn't exist
            ValueError: If required columns are missing
        """
        logger.info(f"Loading test data from: {input_path}")
        
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        try:
            df = pd.read_csv(input_path)
            logger.info(f"Loaded {len(df)} rows from {input_path}")
            
            # Validate required columns
            required_columns = ['id', 'content']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Check for optional columns
            optional_columns = ['book_name', 'char', 'caption']
            present_optional = [col for col in optional_columns if col in df.columns]
            if present_optional:
                logger.info(f"Found optional columns: {present_optional}")
            
            # Log data statistics
            logger.info(f"Data shape: {df.shape}")
            logger.info(f"Columns: {list(df.columns)}")
            
            # Check for missing values
            missing_counts = df.isnull().sum()
            if missing_counts.any():
                logger.warning(f"Missing values found: {missing_counts[missing_counts > 0].to_dict()}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading test data: {e}")
            raise
    
    def retrieve_chunks_for_backstory(self, backstory: str) -> List[str]:
        """
        Retrieve relevant chunks for a given backstory.
        
        Args:
            backstory: The backstory text to search for
            
        Returns:
            List of retrieved chunk texts
        """
        try:
            # Use the existing retrieve_chunks function
            retrieved_df = retrieve_chunks(backstory, k=self.k)
            
            if retrieved_df.empty:
                logger.warning(f"No chunks retrieved for backstory (length: {len(backstory)})")
                self.stats['empty_retrievals'] += 1
                return []
            
            # Extract text from retrieved chunks
            chunks = retrieved_df['text'].tolist()
            
            logger.debug(f"Retrieved {len(chunks)} chunks for backstory")
            return chunks
            
        except Exception as e:
            logger.error(f"Error retrieving chunks: {e}")
            self.stats['failed_predictions'] += 1
            return []
    
    def predict_single_backstory(self, row_id: str, backstory: str) -> Dict[str, Any]:
        """
        Predict consistency for a single backstory.
        
        Args:
            row_id: Unique identifier for the row
            backstory: The backstory text
            
        Returns:
            Dictionary with prediction results
        """
        try:
            # Validate input
            if not isinstance(backstory, str) or pd.isna(backstory):
                logger.warning(f"Empty or invalid backstory for ID {row_id}")
                self.stats['empty_backstories'] += 1
                return {
                    'id': row_id,
                    'prediction': 1,  # Default to consistent for empty backstories
                    'error': 'Empty backstory',
                    'chunks_retrieved': 0
                }
            
            # Retrieve chunks
            retrieved_chunks = self.retrieve_chunks_for_backstory(backstory)
            
            if not retrieved_chunks:
                logger.warning(f"No chunks retrieved for ID {row_id}")
                return {
                    'id': row_id,
                    'prediction': 1,  # Default to consistent when no evidence
                    'error': 'No chunks retrieved',
                    'chunks_retrieved': 0
                }
            
            # Predict consistency
            prediction = predict_consistency(backstory, retrieved_chunks)
            
            self.stats['successful_predictions'] += 1
            
            result = {
                'id': row_id,
                'prediction': prediction,
                'backstory_length': len(backstory),
                'chunks_retrieved': len(retrieved_chunks),
                'error': None
            }
            
            logger.debug(f"ID {row_id}: prediction={prediction}, chunks={len(retrieved_chunks)}")
            return result
            
        except Exception as e:
            logger.error(f"Error processing ID {row_id}: {e}")
            self.stats['failed_predictions'] += 1
            return {
                'id': row_id,
                'prediction': 1,  # Default to consistent on error
                'error': str(e),
                'chunks_retrieved': 0
            }
    
    def process_batch(self, df_batch: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Process a batch of test examples.
        
        Args:
            df_batch: DataFrame batch to process
            
        Returns:
            List of prediction results
        """
        batch_results = []
        
        for idx, row in df_batch.iterrows():
            row_id = str(row['id'])
            backstory = row['content']
            
            logger.info(f"Processing row {idx + 1}/{len(df_batch)} (ID: {row_id})")
            
            result = self.predict_single_backstory(row_id, backstory)
            batch_results.append(result)
            
            self.stats['total_processed'] += 1
            
            # Progress logging
            if self.stats['total_processed'] % 10 == 0:
                logger.info(f"Progress: {self.stats['total_processed']} rows processed")
        
        return batch_results
    
    def run_inference(self, input_path: str) -> List[Dict[str, Any]]:
        """
        Run end-to-end inference on test data.
        
        Args:
            input_path: Path to test.csv file
            
        Returns:
            List of prediction results
        """
        logger.info("Starting end-to-end inference")
        self.stats['start_time'] = time.time()
        
        try:
            # Load test data
            df = self.load_test_data(input_path)
            
            # Process data
            if self.batch_size:
                # Process in batches
                all_results = []
                for i in range(0, len(df), self.batch_size):
                    batch_df = df.iloc[i:i + self.batch_size]
                    logger.info(f"Processing batch {i//self.batch_size + 1}/{(len(df) + self.batch_size - 1)//self.batch_size}")
                    batch_results = self.process_batch(batch_df)
                    all_results.extend(batch_results)
            else:
                # Process all at once (streaming)
                all_results = self.process_batch(df)
            
            self.stats['end_time'] = time.time()
            
            logger.info(f"Inference completed successfully")
            return all_results
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise
    
    def save_results(self, results: List[Dict[str, Any]], output_path: str):
        """
        Save prediction results to CSV file.
        
        Args:
            results: List of prediction results
            output_path: Path to output CSV file
        """
        logger.info(f"Saving results to: {output_path}")
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"Created output directory: {output_dir}")
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Ensure required columns are present and in correct order
        required_columns = ['id', 'prediction']
        for col in required_columns:
            if col not in results_df.columns:
                raise ValueError(f"Missing required column in results: {col}")
        
        # Reorder columns to match specification
        output_columns = ['id', 'prediction']
        additional_columns = [col for col in results_df.columns if col not in output_columns]
        final_columns = output_columns + additional_columns
        
        results_df = results_df[final_columns]
        
        # Save to CSV
        results_df.to_csv(output_path, index=False)
        
        logger.info(f"Results saved: {len(results_df)} rows to {output_path}")
        
        # Log prediction distribution
        prediction_counts = results_df['prediction'].value_counts()
        logger.info(f"Prediction distribution: {prediction_counts.to_dict()}")
    
    def log_statistics(self):
        """Log pipeline execution statistics."""
        if self.stats['start_time'] and self.stats['end_time']:
            duration = self.stats['end_time'] - self.stats['start_time']
            logger.info("=" * 50)
            logger.info("PIPELINE STATISTICS")
            logger.info("=" * 50)
            logger.info(f"Total processed: {self.stats['total_processed']}")
            logger.info(f"Successful: {self.stats['successful_predictions']}")
            logger.info(f"Failed: {self.stats['failed_predictions']}")
            logger.info(f"Empty backstories: {self.stats['empty_backstories']}")
            logger.info(f"Empty retrievals: {self.stats['empty_retrievals']}")
            logger.info(f"Duration: {duration:.2f} seconds")
            
            if self.stats['total_processed'] > 0:
                avg_time = duration / self.stats['total_processed']
                logger.info(f"Average time per row: {avg_time:.3f} seconds")
            
            success_rate = (self.stats['successful_predictions'] / self.stats['total_processed'] * 100) if self.stats['total_processed'] > 0 else 0
            logger.info(f"Success rate: {success_rate:.1f}%")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="End-to-End Prediction Pipeline for Backstory Consistency",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python predict_pipeline.py --input data/test.csv --output results/results.csv
    python predict_pipeline.py --input data/test.csv --output results/results.csv --k 10
    python predict_pipeline.py --input data/test.csv --output results/results.csv --batch-size 50
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Path to test.csv file'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Path to output results.csv file'
    )
    
    parser.add_argument(
        '--k',
        type=int,
        default=5,
        help='Number of chunks to retrieve for each backstory (default: 5)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size for processing (default: process all at once)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    # Parse arguments
    args = parse_arguments()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate arguments
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)
    
    if args.k <= 0:
        logger.error(f"Invalid k value: {args.k}. Must be positive.")
        sys.exit(1)
    
    if args.batch_size and args.batch_size <= 0:
        logger.error(f"Invalid batch size: {args.batch_size}. Must be positive.")
        sys.exit(1)
    
    logger.info("Starting End-to-End Prediction Pipeline")
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")
    logger.info(f"K (chunks to retrieve): {args.k}")
    if args.batch_size:
        logger.info(f"Batch size: {args.batch_size}")
    
    try:
        # Initialize pipeline
        pipeline = PredictionPipeline(k=args.k, batch_size=args.batch_size)
        
        # Run inference
        results = pipeline.run_inference(args.input)
        
        # Save results
        pipeline.save_results(results, args.output)
        
        # Log statistics
        pipeline.log_statistics()
        
        logger.info("Pipeline completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
