#!/usr/bin/env python3
"""
Accuracy Evaluation Script

Tests the accuracy of predictions against ground truth labels for both train and test data.
"""

import argparse
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
import sys
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import prediction function
try:
    from predict_pipeline import PredictionPipeline
    from consistency_final import predict_consistency
    from pathway_final import retrieve_chunks
except ImportError as e:
    print(f"Error importing required modules: {e}")
    sys.exit(1)


class AccuracyEvaluator:
    """Evaluates prediction accuracy against ground truth labels."""
    
    def __init__(self, k: int = 5):
        """Initialize evaluator."""
        self.k = k
        self.results = {}
    
    def load_data(self, file_path: str, require_labels: bool = True) -> pd.DataFrame:
        """Load CSV data with validation."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        df = pd.read_csv(file_path)
        
        # Validate required columns
        if require_labels:
            required_columns = ['id', 'content', 'label']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns in {file_path}: {missing_columns}")
        else:
            required_columns = ['id', 'content']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns in {file_path}: {missing_columns}")
        
        print(f"Loaded {len(df)} rows from {file_path}")
        if require_labels and 'label' in df.columns:
            print(f"Label distribution: {df['label'].value_counts().to_dict()}")
        else:
            print(f"No ground truth labels found in {file_path}")
        
        return df
    
    def load_predictions(self, file_path: str) -> pd.DataFrame:
        """Load prediction results."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Prediction file not found: {file_path}")
        
        df = pd.read_csv(file_path)
        
        # Validate required columns
        required_columns = ['id', 'prediction']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns in {file_path}: {missing_columns}")
        
        print(f"Loaded {len(df)} predictions from {file_path}")
        print(f"Prediction distribution: {df['prediction'].value_counts().to_dict()}")
        
        return df
    
    def generate_predictions(self, data_df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """Generate predictions for dataset."""
        print(f"\nGenerating predictions for {dataset_name} dataset...")
        
        predictions = []
        
        for idx, row in data_df.iterrows():
            row_id = str(row['id'])
            backstory = row['content']
            
            # Retrieve chunks
            try:
                retrieved_df = retrieve_chunks(backstory, k=self.k)
                if retrieved_df.empty:
                    chunks = []
                else:
                    chunks = retrieved_df['text'].tolist()
            except Exception as e:
                print(f"Error retrieving chunks for ID {row_id}: {e}")
                chunks = []
            
            # Predict consistency
            try:
                prediction = predict_consistency(backstory, chunks)
            except Exception as e:
                print(f"Error predicting for ID {row_id}: {e}")
                prediction = 1  # Default to consistent
            
            predictions.append({
                'id': row_id,
                'prediction': prediction,
                'backstory_length': len(backstory),
                'chunks_retrieved': len(chunks)
            })
            
            # Progress
            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1}/{len(data_df)} examples")
        
        return pd.DataFrame(predictions)
    
    def evaluate_dataset(self, data_path: str, predictions_path: str = None, dataset_name: str = "dataset", has_labels: bool = True) -> Dict[str, Any]:
        """Evaluate accuracy for a dataset."""
        print(f"\n{'='*60}")
        print(f"Evaluating {dataset_name.upper()} DATASET")
        print(f"{'='*60}")
        
        # Load ground truth data
        data_df = self.load_data(data_path, require_labels=has_labels)
        
        # Load or generate predictions
        if predictions_path and os.path.exists(predictions_path):
            predictions_df = self.load_predictions(predictions_path)
            print(f"Using existing predictions from {predictions_path}")
        else:
            predictions_df = self.generate_predictions(data_df, dataset_name)
            # Save predictions for future use
            output_path = f"results/{dataset_name}_predictions.csv"
            os.makedirs('results', exist_ok=True)
            predictions_df.to_csv(output_path, index=False)
            print(f"Saved predictions to {output_path}")
        
        if has_labels:
            # Merge predictions with ground truth
            data_df['id'] = data_df['id'].astype(str)
            predictions_df['id'] = predictions_df['id'].astype(str)
            
            merged_df = data_df[['id', 'label']].merge(
                predictions_df[['id', 'prediction']], 
                on='id', 
                how='inner'
            )
            
            if len(merged_df) != len(data_df):
                print(f"Warning: Merged {len(merged_df)} rows, expected {len(data_df)}")
            
            # Calculate metrics
            # Convert string labels to numeric for comparison
            label_mapping = {'contradict': 0, 'consistent': 1}
            y_true = merged_df['label'].map(label_mapping)
            y_pred = merged_df['prediction']
            
            # Handle any unmapped labels
            if y_true.isnull().any():
                print(f"Warning: Some labels could not be mapped: {merged_df['label'].unique()}")
                y_true = y_true.fillna(1)  # Default to consistent
            
            accuracy = accuracy_score(y_true, y_pred)
            
            # Detailed classification report
            class_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            
            # Confusion matrix
            conf_matrix = confusion_matrix(y_true, y_pred)
            
            results = {
                'dataset': dataset_name,
                'total_examples': len(merged_df),
                'accuracy': accuracy,
                'correct_predictions': int(accuracy * len(merged_df)),
                'incorrect_predictions': len(merged_df) - int(accuracy * len(merged_df)),
                'label_distribution': data_df['label'].value_counts().to_dict(),
                'prediction_distribution': predictions_df['prediction'].value_counts().to_dict(),
                'classification_report': class_report,
                'confusion_matrix': conf_matrix,
                'precision_0': class_report['0']['precision'] if '0' in class_report else 0,
                'precision_1': class_report['1']['precision'] if '1' in class_report else 0,
                'recall_0': class_report['0']['recall'] if '0' in class_report else 0,
                'recall_1': class_report['1']['recall'] if '1' in class_report else 0,
                'f1_0': class_report['0']['f1-score'] if '0' in class_report else 0,
                'f1_1': class_report['1']['f1-score'] if '1' in class_report else 0,
            }
        else:
            # No ground truth labels available
            results = {
                'dataset': dataset_name,
                'total_examples': len(data_df),
                'prediction_distribution': predictions_df['prediction'].value_counts().to_dict(),
                'note': 'No ground truth labels available for accuracy calculation'
            }
        
        self.results[dataset_name] = results
        return results
    
    def print_results(self, results: Dict[str, Any]):
        """Print detailed evaluation results."""
        print(f"\n{'='*80}")
        print("ACCURACY EVALUATION RESULTS")
        print(f"{'='*80}")
        
        for dataset_name, metrics in results.items():
            print(f"\n📊 {dataset_name.upper()} DATASET RESULTS:")
            print(f"{'-'*50}")
            print(f"Total Examples: {metrics['total_examples']}")
            
            if 'accuracy' in metrics:
                # Dataset with ground truth labels
                print(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
                print(f"Correct: {metrics['correct_predictions']}")
                print(f"Incorrect: {metrics['incorrect_predictions']}")
                
                print(f"\n📈 Label Distribution (Ground Truth):")
                for label, count in metrics['label_distribution'].items():
                    print(f"  Label {label}: {count} ({count/metrics['total_examples']*100:.1f}%)")
                
                print(f"\n🎯 Prediction Distribution:")
                for pred, count in metrics['prediction_distribution'].items():
                    print(f"  Prediction {pred}: {count} ({count/metrics['total_examples']*100:.1f}%)")
                
                print(f"\n📋 Detailed Metrics:")
                print(f"  Precision (Class 0): {metrics['precision_0']:.4f}")
                print(f"  Precision (Class 1): {metrics['precision_1']:.4f}")
                print(f"  Recall (Class 0): {metrics['recall_0']:.4f}")
                print(f"  Recall (Class 1): {metrics['recall_1']:.4f}")
                print(f"  F1-Score (Class 0): {metrics['f1_0']:.4f}")
                print(f"  F1-Score (Class 1): {metrics['f1_1']:.4f}")
                
                print(f"\n🔢 Confusion Matrix:")
                print(f"  Predicted\\Actual    0    1")
                print(f"  0                {metrics['confusion_matrix'][0][0]:4d}    {metrics['confusion_matrix'][0][1]:4d}")
                print(f"  1                {metrics['confusion_matrix'][1][0]:4d}    {metrics['confusion_matrix'][1][1]:4d}")
            else:
                # Dataset without ground truth labels
                print(f"Note: {metrics.get('note', 'No ground truth labels available')}")
                
                print(f"\n🎯 Prediction Distribution:")
                for pred, count in metrics['prediction_distribution'].items():
                    print(f"  Prediction {pred}: {count} ({count/metrics['total_examples']*100:.1f}%)")
    
    def save_results(self, results: Dict[str, Any], output_path: str = "results/accuracy_evaluation.txt"):
        """Save evaluation results to file."""
        os.makedirs('results', exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write("ACCURACY EVALUATION RESULTS\n")
            f.write("="*50 + "\n\n")
            
            for dataset_name, metrics in results.items():
                f.write(f"{dataset_name.upper()} DATASET\n")
                f.write("-"*30 + "\n")
                f.write(f"Total Examples: {metrics['total_examples']}\n")
                
                if 'accuracy' in metrics:
                    f.write(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)\n")
                    f.write(f"Correct: {metrics['correct_predictions']}\n")
                    f.write(f"Incorrect: {metrics['incorrect_predictions']}\n\n")
                    
                    f.write("Label Distribution (Ground Truth):\n")
                    for label, count in metrics['label_distribution'].items():
                        f.write(f"  Label {label}: {count} ({count/metrics['total_examples']*100:.1f}%)\n")
                    
                    f.write("\nPrediction Distribution:\n")
                    for pred, count in metrics['prediction_distribution'].items():
                        f.write(f"  Prediction {pred}: {count} ({count/metrics['total_examples']*100:.1f}%)\n")
                    
                    f.write("\nDetailed Metrics:\n")
                    f.write(f"  Precision (Class 0): {metrics['precision_0']:.4f}\n")
                    f.write(f"  Precision (Class 1): {metrics['precision_1']:.4f}\n")
                    f.write(f"  Recall (Class 0): {metrics['recall_0']:.4f}\n")
                    f.write(f"  Recall (Class 1): {metrics['recall_1']:.4f}\n")
                    f.write(f"  F1-Score (Class 0): {metrics['f1_0']:.4f}\n")
                    f.write(f"  F1-Score (Class 1): {metrics['f1_1']:.4f}\n")
                    
                    f.write("\nConfusion Matrix:\n")
                    f.write("Predicted\\Actual    0    1\n")
                    f.write(f"0                {metrics['confusion_matrix'][0][0]:4d}    {metrics['confusion_matrix'][0][1]:4d}\n")
                    f.write(f"1                {metrics['confusion_matrix'][1][0]:4d}    {metrics['confusion_matrix'][1][1]:4d}\n")
                else:
                    f.write(f"Note: {metrics.get('note', 'No ground truth labels available')}\n")
                    f.write("\nPrediction Distribution:\n")
                    for pred, count in metrics['prediction_distribution'].items():
                        f.write(f"  Prediction {pred}: {count} ({count/metrics['total_examples']*100:.1f}%)\n")
                
                f.write("\n" + "="*50 + "\n\n")
        
        print(f"\n💾 Results saved to {output_path}")
    
    def plot_confusion_matrices(self, results: Dict[str, Any]):
        """Plot confusion matrices for visual comparison."""
        try:
            fig, axes = plt.subplots(1, len(results), figsize=(12, 5))
            
            for idx, (dataset_name, metrics) in enumerate(results.items()):
                ax = axes[idx] if len(results) > 1 else axes
                
                sns.heatmap(
                    metrics['confusion_matrix'], 
                    annot=True, 
                    fmt='d', 
                    cmap='Blues',
                    ax=ax,
                    cbar=False
                )
                ax.set_title(f'{dataset_name.title()} Dataset\nAccuracy: {metrics["accuracy"]:.3f}')
                ax.set_xlabel('Predicted Label')
                ax.set_ylabel('True Label')
                ax.set_xticklabels(['0 (Inconsistent)', '1 (Consistent)'])
                ax.set_yticklabels(['0 (Inconsistent)', '1 (Consistent)'])
            
            plt.tight_layout()
            plt.savefig('results/confusion_matrices.png', dpi=300, bbox_inches='tight')
            print(f"\n📊 Confusion matrices saved to results/confusion_matrices.png")
            
        except Exception as e:
            print(f"Error plotting confusion matrices: {e}")
    
    def compare_datasets(self, results: Dict[str, Any]):
        """Compare performance between datasets."""
        if len(results) < 2:
            print(f"\n Note: Only {len(results)} dataset(s) evaluated")
            return
        
        print(f"\n DATASET COMPARISON")
        print(f"{'-'*50}")
        
        datasets = list(results.keys())
        metrics_comparison = []
        
        for dataset in datasets:
            metrics = results[dataset]
            comparison_row = {
                'Dataset': dataset.title(),
                'Total': metrics['total_examples'],
                'Prediction Distribution': str(metrics['prediction_distribution'])
            }
            
            # Add accuracy metrics if available
            if 'accuracy' in metrics:
                comparison_row.update({
                    'Accuracy': f"{metrics['accuracy']:.4f}",
                    'Precision_0': f"{metrics['precision_0']:.4f}",
                    'Precision_1': f"{metrics['precision_1']:.4f}",
                    'Recall_0': f"{metrics['recall_0']:.4f}",
                    'Recall_1': f"{metrics['recall_1']:.4f}",
                    'F1_0': f"{metrics['f1_0']:.4f}",
                    'F1_1': f"{metrics['f1_1']:.4f}",
                })
            else:
                comparison_row.update({
                    'Accuracy': 'N/A (no labels)',
                    'Precision_0': 'N/A',
                    'Precision_1': 'N/A',
                    'Recall_0': 'N/A',
                    'Recall_1': 'N/A',
                    'F1_0': 'N/A',
                    'F1_1': 'N/A',
                })
            
            metrics_comparison.append(comparison_row)
        
        comparison_df = pd.DataFrame(metrics_comparison)
        print(comparison_df.to_string(index=False))
        
        # Save comparison
        comparison_df.to_csv('results/dataset_comparison.csv', index=False)
        print(f"\n Comparison saved to results/dataset_comparison.csv")
        print(f"\n💾 Comparison saved to results/dataset_comparison.csv")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Accuracy Evaluation for Train and Test Datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python evaluate_accuracy.py --train data/train.csv --test data/test.csv
    python evaluate_accuracy.py --train data/train.csv --test data/test.csv --test-predictions results/results.csv
    python evaluate_accuracy.py --train data/train.csv --test data/test.csv --k 10
        """
    )
    
    parser.add_argument(
        '--train',
        type=str,
        required=True,
        help='Path to train.csv file with ground truth labels'
    )
    
    parser.add_argument(
        '--test',
        type=str,
        required=True,
        help='Path to test.csv file with ground truth labels'
    )
    
    parser.add_argument(
        '--test-predictions',
        type=str,
        default=None,
        help='Path to existing test predictions (optional, will generate if not provided)'
    )
    
    parser.add_argument(
        '--k',
        type=int,
        default=5,
        help='Number of chunks to retrieve for each backstory (default: 5)'
    )
    
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Generate confusion matrix plots'
    )
    
    return parser.parse_args()


def main():
    """Main evaluation function."""
    args = parse_arguments()
    
    print("🎯 ACCURACY EVALUATION TOOL")
    print("="*50)
    print(f"Train data: {args.train}")
    print(f"Test data: {args.test}")
    print(f"Chunks to retrieve (k): {args.k}")
    if args.test_predictions:
        print(f"Using existing test predictions: {args.test_predictions}")
    print("="*50)
    
    try:
        # Initialize evaluator
        evaluator = AccuracyEvaluator(k=args.k)
        
        # Evaluate train dataset
        train_results = evaluator.evaluate_dataset(
            args.train, 
            dataset_name="train",
            has_labels=True
        )
        
        # Evaluate test dataset
        test_results = evaluator.evaluate_dataset(
            args.test,
            predictions_path=args.test_predictions,
            dataset_name="test",
            has_labels=False  # Test data typically doesn't have labels
        )
        
        # Combine results
        all_results = {
            'train': train_results,
            'test': test_results
        }
        
        # Print results
        evaluator.print_results(all_results)
        
        # Save results
        evaluator.save_results(all_results)
        
        # Generate plots if requested
        if args.plot:
            evaluator.plot_confusion_matrices(all_results)
        
        # Compare datasets
        evaluator.compare_datasets(all_results)
        
        print(f"\n🎉 EVALUATION COMPLETED SUCCESSFULLY!")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
