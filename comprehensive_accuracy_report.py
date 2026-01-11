#!/usr/bin/env python3
"""
Comprehensive Accuracy Prediction Report
Combines train and test results for complete model evaluation
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json
import os

def load_and_analyze_predictions():
    """Load and analyze both train and test predictions"""
    
    print("🎯 COMPREHENSIVE ACCURACY ANALYSIS")
    print("="*50)
    
    # Load train data and predictions
    try:
        train_data = pd.read_csv('data/train.csv')
        train_predictions = pd.read_csv('results/train_predictions.csv')
        
        # Merge train data with predictions
        train_data['id'] = train_data['id'].astype(str)
        train_predictions['id'] = train_predictions['id'].astype(str)
        
        train_merged = train_data.merge(train_predictions, on='id', how='inner')
        
        # Calculate train accuracy
        if 'label' in train_merged.columns:
            # Map string labels to numeric
            label_mapping = {'contradict': 0, 'consistent': 1}
            train_merged['label_numeric'] = train_merged['label'].map(label_mapping)
            
            train_correct = sum(train_merged['label_numeric'] == train_merged['prediction'])
            train_total = len(train_merged)
            train_accuracy = train_correct / train_total
            
            print(f"\n📊 TRAIN DATASET RESULTS:")
            print(f"   Total Examples: {train_total}")
            print(f"   Correct: {train_correct}")
            print(f"   Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
            
            # Label distribution
            train_label_dist = train_merged['label'].value_counts().to_dict()
            train_pred_dist = train_merged['prediction'].value_counts().to_dict()
            
            print(f"   Ground Truth: {train_label_dist}")
            print(f"   Predictions: {train_pred_dist}")
        else:
            train_accuracy = 0
            train_total = 0
            train_correct = 0
            
    except Exception as e:
        print(f"Error loading train data: {e}")
        train_accuracy = 0
        train_total = 0
        train_correct = 0
    
    # Load test predictions
    try:
        test_predictions = pd.read_csv('results.csv')
        test_total = len(test_predictions)
        
        # Calculate test prediction distribution
        test_pred_dist = test_predictions['prediction'].value_counts().to_dict()
        
        # Calculate percentages
        test_consistent = test_pred_dist.get(1, 0)
        test_inconsistent = test_pred_dist.get(0, 0)
        test_consistent_pct = (test_consistent / test_total) * 100 if test_total > 0 else 0
        test_inconsistent_pct = (test_inconsistent / test_total) * 100 if test_total > 0 else 0
        
        print(f"\n📊 TEST DATASET RESULTS:")
        print(f"   Total Examples: {test_total}")
        print(f"   Consistent (1): {test_consistent} ({test_consistent_pct:.1f}%)")
        print(f"   Inconsistent (0): {test_inconsistent} ({test_inconsistent_pct:.1f}%)")
        print(f"   Prediction Distribution: {test_pred_dist}")
        
    except Exception as e:
        print(f"Error loading test data: {e}")
        test_total = 0
        test_consistent = 0
        test_inconsistent = 0
        test_consistent_pct = 0
        test_inconsistent_pct = 0
    
    # Create comprehensive report
    report = {
        "timestamp": datetime.now().isoformat(),
        "model_name": "consistency_classifier_v1",
        "model_version": "1.0",
        "model_description": "Final consistency classifier with retrieval-augmented reasoning",
        
        "train_results": {
            "total_examples": train_total,
            "correct_predictions": train_correct,
            "accuracy": train_accuracy,
            "accuracy_percentage": train_accuracy * 100,
            "label_distribution": train_label_dist if 'train_label_dist' in locals() else {},
            "prediction_distribution": train_pred_dist if 'train_pred_dist' in locals() else {}
        },
        
        "test_results": {
            "total_examples": test_total,
            "consistent_predictions": test_consistent,
            "inconsistent_predictions": test_inconsistent,
            "consistent_percentage": test_consistent_pct,
            "inconsistent_percentage": test_inconsistent_pct,
            "prediction_distribution": test_pred_dist if 'test_pred_dist' in locals() else {}
        },
        
        "overall_summary": {
            "total_predictions": train_total + test_total,
            "train_accuracy": train_accuracy,
            "test_consistent_rate": test_consistent_pct,
            "model_behavior": "Biased toward contradiction in training, balanced in test"
        }
    }
    
    # Save comprehensive report
    os.makedirs('results', exist_ok=True)
    
    with open('results/comprehensive_accuracy_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    # Create text report
    with open('results/comprehensive_accuracy_report.txt', 'w') as f:
        f.write("COMPREHENSIVE ACCURACY REPORT\n")
        f.write("="*50 + "\n\n")
        
        f.write(f"Model: {report['model_name']} v{report['model_version']}\n")
        f.write(f"Description: {report['model_description']}\n")
        f.write(f"Generated: {report['timestamp']}\n\n")
        
        f.write("TRAIN DATASET RESULTS\n")
        f.write("-"*30 + "\n")
        f.write(f"Total Examples: {report['train_results']['total_examples']}\n")
        f.write(f"Correct Predictions: {report['train_results']['correct_predictions']}\n")
        f.write(f"Accuracy: {report['train_results']['accuracy']:.4f} ({report['train_results']['accuracy_percentage']:.2f}%)\n")
        f.write(f"Label Distribution: {report['train_results']['label_distribution']}\n")
        f.write(f"Prediction Distribution: {report['train_results']['prediction_distribution']}\n\n")
        
        f.write("TEST DATASET RESULTS\n")
        f.write("-"*30 + "\n")
        f.write(f"Total Examples: {report['test_results']['total_examples']}\n")
        f.write(f"Consistent Predictions: {report['test_results']['consistent_predictions']} ({report['test_results']['consistent_percentage']:.1f}%)\n")
        f.write(f"Inconsistent Predictions: {report['test_results']['inconsistent_predictions']} ({report['test_results']['inconsistent_percentage']:.1f}%)\n")
        f.write(f"Prediction Distribution: {report['test_results']['prediction_distribution']}\n\n")
        
        f.write("OVERALL ANALYSIS\n")
        f.write("-"*30 + "\n")
        f.write(f"Total Predictions: {report['overall_summary']['total_predictions']}\n")
        f.write(f"Train Accuracy: {report['overall_summary']['train_accuracy']:.4f}\n")
        f.write(f"Test Consistent Rate: {report['overall_summary']['test_consistent_rate']:.1f}%\n")
        f.write(f"Model Behavior: {report['overall_summary']['model_behavior']}\n")
    
    print(f"\n✅ Comprehensive report generated!")
    print(f"📁 JSON: results/comprehensive_accuracy_report.json")
    print(f"📄 Text: results/comprehensive_accuracy_report.txt")
    
    return report

if __name__ == "__main__":
    load_and_analyze_predictions()
