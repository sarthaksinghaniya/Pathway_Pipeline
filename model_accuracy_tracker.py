#!/usr/bin/env python3
"""
Model Accuracy Prediction and Storage

Predicts accuracy of results produced by ML models and stores accuracy metrics.
Tracks model performance over time for analysis and comparison.
"""

import argparse
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import sys
import os
from datetime import datetime
import json
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import evaluation functions
try:
    from evaluate_accuracy import AccuracyEvaluator
except ImportError as e:
    print(f"Error importing evaluation functions: {e}")
    sys.exit(1)


class ModelAccuracyTracker:
    """Tracks and predicts model accuracy over time."""
    
    def __init__(self, results_dir: str = "results"):
        """Initialize tracker."""
        self.results_dir = results_dir
        self.accuracy_history = []
        self.models_data = {}
        
        # Ensure results directory exists
        os.makedirs(results_dir, exist_ok=True)
        
        # Load existing history if available
        self.load_accuracy_history()
    
    def load_accuracy_history(self):
        """Load existing accuracy history."""
        history_file = os.path.join(self.results_dir, "model_accuracy_history.json")
        
        if os.path.exists(history_file):
            try:
                with open(history_file, 'r') as f:
                    self.accuracy_history = json.load(f)
                print(f"Loaded {len(self.accuracy_history)} previous accuracy records")
            except Exception as e:
                print(f"Error loading accuracy history: {e}")
                self.accuracy_history = []
        else:
            print("No existing accuracy history found")
    
    def save_accuracy_history(self):
        """Save accuracy history to file."""
        history_file = os.path.join(self.results_dir, "model_accuracy_history.json")
        
        try:
            with open(history_file, 'w') as f:
                json.dump(self.accuracy_history, f, indent=2, default=str)
            print(f"Saved accuracy history with {len(self.accuracy_history)} records")
        except Exception as e:
            print(f"Error saving accuracy history: {e}")
    
    def predict_model_accuracy(self, 
                          predictions_file: str, 
                          ground_truth_file: str,
                          model_name: str = "current_model",
                          model_version: str = "1.0",
                          model_description: str = "") -> Dict[str, Any]:
        """
        Predict accuracy for a model's predictions.
        
        Args:
            predictions_file: Path to model predictions CSV
            ground_truth_file: Path to ground truth labels CSV
            model_name: Name of the model
            model_version: Version of the model
            model_description: Description of model changes
            
        Returns:
            Dictionary with accuracy metrics
        """
        print(f"\n🔍 Predicting accuracy for model: {model_name} v{model_version}")
        
        # Load predictions
        if not os.path.exists(predictions_file):
            raise FileNotFoundError(f"Predictions file not found: {predictions_file}")
        
        predictions_df = pd.read_csv(predictions_file)
        print(f"Loaded {len(predictions_df)} predictions from {predictions_file}")
        
        # Load ground truth
        if not os.path.exists(ground_truth_file):
            raise FileNotFoundError(f"Ground truth file not found: {ground_truth_file}")
        
        truth_df = pd.read_csv(ground_truth_file)
        print(f"Loaded {len(truth_df)} ground truth examples from {ground_truth_file}")
        
        # Validate required columns
        pred_cols = ['id', 'prediction']
        truth_cols = ['id', 'label']
        
        missing_pred = [col for col in pred_cols if col not in predictions_df.columns]
        missing_truth = [col for col in truth_cols if col not in truth_df.columns]
        
        if missing_pred:
            raise ValueError(f"Missing columns in predictions: {missing_pred}")
        if missing_truth:
            raise ValueError(f"Missing columns in ground truth: {missing_truth}")
        
        # Merge predictions with ground truth
        predictions_df['id'] = predictions_df['id'].astype(str)
        truth_df['id'] = truth_df['id'].astype(str)
        
        merged_df = predictions_df.merge(truth_df, on='id', how='inner')
        
        if len(merged_df) == 0:
            print(f"Warning: No matching IDs found between predictions and ground truth")
            print(f"Predictions IDs: {predictions_df['id'].tolist()[:10]}...")
            print(f"Ground Truth IDs: {truth_df['id'].tolist()[:10]}...")
            return None
        
        if len(merged_df) != min(len(predictions_df), len(truth_df)):
            print(f"Warning: Merged {len(merged_df)} rows, predictions: {len(predictions_df)}, ground truth: {len(truth_df)}")
        
        # Calculate accuracy metrics
        # Handle string labels if present
        if merged_df['label'].dtype == 'object':
            label_mapping = {'contradict': 0, 'consistent': 1}
            y_true = merged_df['label'].map(label_mapping)
            if y_true.isnull().any():
                print(f"Warning: Some labels could not be mapped: {merged_df['label'].unique()}")
                y_true = y_true.fillna(1)
        else:
            y_true = merged_df['label']
        
        y_pred = merged_df['prediction']
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        class_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        conf_matrix = confusion_matrix(y_true, y_pred)
        
        # Create accuracy record
        accuracy_record = {
            'timestamp': datetime.now().isoformat(),
            'model_name': model_name,
            'model_version': model_version,
            'model_description': model_description,
            'predictions_file': predictions_file,
            'ground_truth_file': ground_truth_file,
            'total_examples': len(merged_df),
            'correct_predictions': int(accuracy * len(merged_df)),
            'incorrect_predictions': len(merged_df) - int(accuracy * len(merged_df)),
            'accuracy': accuracy,
            'accuracy_percentage': accuracy * 100,
            'precision_0': class_report['0']['precision'] if '0' in class_report else 0,
            'precision_1': class_report['1']['precision'] if '1' in class_report else 0,
            'recall_0': class_report['0']['recall'] if '0' in class_report else 0,
            'recall_1': class_report['1']['recall'] if '1' in class_report else 0,
            'f1_0': class_report['0']['f1-score'] if '0' in class_report else 0,
            'f1_1': class_report['1']['f1-score'] if '1' in class_report else 0,
            'confusion_matrix': conf_matrix.tolist(),
            'label_distribution': merged_df['label'].value_counts().to_dict() if 'label' in merged_df.columns else {},
            'prediction_distribution': merged_df['prediction'].value_counts().to_dict()
        }
        
        # Add to history
        self.accuracy_history.append(accuracy_record)
        
        # Save updated history
        self.save_accuracy_history()
        
        print(f"\n✅ Accuracy Prediction Complete:")
        print(f"   Model: {model_name} v{model_version}")
        print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   Correct: {accuracy_record['correct_predictions']}/{accuracy_record['total_examples']}")
        print(f"   Total Examples: {accuracy_record['total_examples']}")
        
        return accuracy_record
    
    def compare_models(self, model_names: Optional[List[str]] = None) -> pd.DataFrame:
        """Compare performance of different models."""
        if not self.accuracy_history:
            print("No accuracy history available for comparison")
            return pd.DataFrame()
        
        # Filter by model names if specified
        if model_names:
            filtered_history = [
                record for record in self.accuracy_history 
                if record['model_name'] in model_names
            ]
        else:
            filtered_history = self.accuracy_history
        
        if not filtered_history:
            print("No matching models found in history")
            return pd.DataFrame()
        
        # Get latest record for each model
        latest_records = {}
        for record in filtered_history:
            model_name = record['model_name']
            if model_name not in latest_records:
                latest_records[model_name] = record
            else:
                # Compare timestamps
                current_time = datetime.fromisoformat(record['timestamp'])
                latest_time = datetime.fromisoformat(latest_records[model_name]['timestamp'])
                if current_time >= latest_time:
                    latest_records[model_name] = record
        
        # Create comparison DataFrame
        comparison_data = []
        for model_name, record in latest_records.items():
            comparison_data.append({
                'Model': model_name,
                'Version': record['model_version'],
                'Accuracy': f"{record['accuracy']:.4f}",
                'Accuracy %': f"{record['accuracy']*100:.2f}%",
                'Total Examples': record['total_examples'],
                'Correct': record['correct_predictions'],
                'Date': record['timestamp'][:10]  # YYYY-MM-DD format
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
        
        print("\n📊 Model Comparison:")
        print(comparison_df.to_string(index=False))
        
        # Save comparison
        comparison_file = os.path.join(self.results_dir, "model_comparison.csv")
        comparison_df.to_csv(comparison_file, index=False)
        print(f"\n💾 Model comparison saved to {comparison_file}")
        
        return comparison_df
    
    def plot_accuracy_trend(self, model_name: Optional[str] = None):
        """Plot accuracy trend over time."""
        if not self.accuracy_history:
            print("No accuracy history available for plotting")
            return
        
        # Filter by model name if specified
        if model_name:
            filtered_history = [
                record for record in self.accuracy_history 
                if record['model_name'] == model_name
            ]
            if not filtered_history:
                print(f"No accuracy history found for model: {model_name}")
                return
        else:
            filtered_history = self.accuracy_history
        
        if not filtered_history:
            print("No data available for plotting")
            return
        
        # Prepare data for plotting
        df = pd.DataFrame(filtered_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Create plot
        plt.figure(figsize=(12, 6))
        
        if model_name:
            # Single model trend
            plt.plot(df['timestamp'], df['accuracy'], marker='o', linewidth=2, markersize=8)
            plt.title(f'Accuracy Trend for {model_name}')
            plt.ylabel('Accuracy')
            plt.xlabel('Date')
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
        else:
            # Multiple models comparison
            for model in df['model_name'].unique():
                model_data = df[df['model_name'] == model]
                plt.plot(model_data['timestamp'], model_data['accuracy'], 
                        marker='o', linewidth=2, markersize=6, label=model)
            
            plt.title('Model Accuracy Comparison Over Time')
            plt.ylabel('Accuracy')
            plt.xlabel('Date')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = os.path.join(self.results_dir, "accuracy_trend.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"\n📊 Accuracy trend plot saved to {plot_file}")
        
        plt.show()
    
    def generate_accuracy_report(self, output_file: Optional[str] = None):
        """Generate comprehensive accuracy report."""
        if not self.accuracy_history:
            print("No accuracy history available for report generation")
            return
        
        if not output_file:
            output_file = os.path.join(self.results_dir, "accuracy_report.txt")
        
        # Create report
        with open(output_file, 'w') as f:
            f.write("MODEL ACCURACY REPORT\n")
            f.write("="*60 + "\n\n")
            
            # Summary statistics
            if len(self.accuracy_history) > 0:
                accuracies = [record['accuracy'] for record in self.accuracy_history]
                f.write(f"Total Models Evaluated: {len(set(record['model_name'] for record in self.accuracy_history))}\n")
                f.write(f"Total Evaluations: {len(self.accuracy_history)}\n")
                f.write(f"Best Accuracy: {max(accuracies):.4f}\n")
                f.write(f"Worst Accuracy: {min(accuracies):.4f}\n")
                f.write(f"Average Accuracy: {np.mean(accuracies):.4f}\n")
                f.write(f"Accuracy Std Dev: {np.std(accuracies):.4f}\n\n")
            
            # Detailed records
            f.write("DETAILED EVALUATION HISTORY\n")
            f.write("-"*40 + "\n")
            
            for i, record in enumerate(self.accuracy_history, 1):
                f.write(f"Evaluation #{i}\n")
                f.write(f"Date: {record['timestamp']}\n")
                f.write(f"Model: {record['model_name']} v{record['model_version']}\n")
                if record['model_description']:
                    f.write(f"Description: {record['model_description']}\n")
                f.write(f"Predictions: {record['predictions_file']}\n")
                f.write(f"Ground Truth: {record['ground_truth_file']}\n")
                f.write(f"Total Examples: {record['total_examples']}\n")
                f.write(f"Accuracy: {record['accuracy']:.4f} ({record['accuracy']*100:.2f}%)\n")
                f.write(f"Correct: {record['correct_predictions']}\n")
                f.write(f"Incorrect: {record['incorrect_predictions']}\n")
                f.write(f"Precision (0): {record['precision_0']:.4f}\n")
                f.write(f"Precision (1): {record['precision_1']:.4f}\n")
                f.write(f"Recall (0): {record['recall_0']:.4f}\n")
                f.write(f"Recall (1): {record['recall_1']:.4f}\n")
                f.write(f"F1-Score (0): {record['f1_0']:.4f}\n")
                f.write(f"F1-Score (1): {record['f1_1']:.4f}\n")
                f.write("Label Distribution: " + str(record['label_distribution']) + "\n")
                f.write("Prediction Distribution: " + str(record['prediction_distribution']) + "\n")
                f.write("+"*60 + "\n\n")
        
        print(f"\n💾 Comprehensive report saved to {output_file}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Model Accuracy Prediction and Storage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python model_accuracy_tracker.py --predictions results/results.csv --ground-truth data/train.csv
    python model_accuracy_tracker.py --predictions results/results.csv --ground-truth data/train.csv --model-name consistency_v1
    python model_accuracy_tracker.py --compare-models consistency_v1,consistency_v2
    python model_accuracy_tracker.py --plot-trend --model-name consistency_v1
        """
    )
    
    parser.add_argument(
        '--predictions', '-p',
        type=str,
        required=True,
        help='Path to model predictions CSV file'
    )
    
    parser.add_argument(
        '--ground-truth', '-g',
        type=str,
        required=True,
        help='Path to ground truth labels CSV file'
    )
    
    parser.add_argument(
        '--model-name', '-m',
        type=str,
        default='current_model',
        help='Name of the model (default: current_model)'
    )
    
    parser.add_argument(
        '--model-version', '-v',
        type=str,
        default='1.0',
        help='Version of the model (default: 1.0)'
    )
    
    parser.add_argument(
        '--model-description', '-d',
        type=str,
        default='',
        help='Description of model changes or configuration'
    )
    
    parser.add_argument(
        '--compare-models',
        nargs='+',
        help='Compare specific models from history'
    )
    
    parser.add_argument(
        '--plot-trend',
        action='store_true',
        help='Plot accuracy trend over time'
    )
    
    parser.add_argument(
        '--plot-model',
        type=str,
        help='Plot trend for specific model only'
    )
    
    parser.add_argument(
        '--generate-report',
        action='store_true',
        help='Generate comprehensive accuracy report'
    )
    
    parser.add_argument(
        '--results-dir',
        type=str,
        default='results',
        help='Directory to store results (default: results)'
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_arguments()
    
    print("🎯 MODEL ACCURACY TRACKER")
    print("="*50)
    
    try:
        # Initialize tracker
        tracker = ModelAccuracyTracker(results_dir=args.results_dir)
        
        # Predict accuracy for current model
        accuracy_record = tracker.predict_model_accuracy(
            predictions_file=args.predictions,
            ground_truth_file=args.ground_truth,
            model_name=args.model_name,
            model_version=args.model_version,
            model_description=args.model_description
        )
        
        # Compare models if requested
        if args.compare_models:
            tracker.compare_models(args.compare_models)
        
        # Plot trends if requested
        if args.plot_trend:
            tracker.plot_accuracy_trend()
        elif args.plot_model:
            tracker.plot_accuracy_trend(model_name=args.plot_model)
        
        # Generate report if requested
        if args.generate_report:
            tracker.generate_accuracy_report()
        
        print(f"\n🎉 ACCURACY TRACKING COMPLETED!")
        print(f"Results stored in: {args.results_dir}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
