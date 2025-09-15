"""
Fairness Analysis Module for NLP Applications

This module provides comprehensive fairness analysis tools for NLP models,
including bias detection, fairness metrics, and visualization capabilities.
"""

import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from pathlib import Path
import json
import pandas as pd
import numpy as np
from transformers import (
    pipeline, 
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    Pipeline
)
from datasets import Dataset
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


@dataclass
class FairnessMetrics:
    """Container for fairness analysis metrics."""
    
    demographic_parity: float
    equalized_odds: float
    equal_opportunity: float
    statistical_parity: float
    accuracy_by_group: Dict[str, float]
    precision_by_group: Dict[str, float]
    recall_by_group: Dict[str, float]
    f1_by_group: Dict[str, float]
    bias_score: float
    overall_accuracy: float


@dataclass
class BiasTestResult:
    """Container for bias test results."""
    
    test_name: str
    is_biased: bool
    bias_score: float
    confidence: float
    details: Dict[str, Any]


class FairnessAnalyzer:
    """
    Comprehensive fairness analyzer for NLP models.
    
    This class provides methods to analyze model fairness across different
    demographic groups and identify potential biases in predictions.
    """
    
    def __init__(
        self, 
        model_name: str = "bert-base-uncased",
        device: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize the fairness analyzer.
        
        Args:
            model_name: Name of the pre-trained model to use
            device: Device to run the model on ('cpu', 'cuda', etc.)
            cache_dir: Directory to cache model files
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_dir = cache_dir
        
        # Initialize model components
        self.classifier: Optional[Pipeline] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModelForSequenceClassification] = None
        
        # Analysis results storage
        self.results: Dict[str, Any] = {}
        self.bias_tests: List[BiasTestResult] = []
        
        logger.info(f"Initialized FairnessAnalyzer with model: {model_name}")
    
    def load_model(self) -> None:
        """Load the pre-trained model and tokenizer."""
        try:
            logger.info(f"Loading model: {self.model_name}")
            
            # Load classifier pipeline
            self.classifier = pipeline(
                "text-classification",
                model=self.model_name,
                device=self.device,
                cache_dir=self.cache_dir
            )
            
            # Load tokenizer and model separately for advanced analysis
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def analyze_text_bias(
        self, 
        texts: List[str], 
        sensitive_attributes: List[str],
        labels: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Analyze bias in text predictions.
        
        Args:
            texts: List of input texts to analyze
            sensitive_attributes: List of sensitive attributes to check for bias
            labels: Optional ground truth labels
            
        Returns:
            Dictionary containing bias analysis results
        """
        if self.classifier is None:
            self.load_model()
        
        logger.info(f"Analyzing bias in {len(texts)} texts")
        
        # Get predictions
        predictions = self.classifier(texts)
        
        # Extract prediction labels and scores
        pred_labels = [pred[0]['label'] for pred in predictions]
        pred_scores = [pred[0]['score'] for pred in predictions]
        
        # Analyze bias patterns
        bias_analysis = self._analyze_bias_patterns(
            texts, pred_labels, pred_scores, sensitive_attributes, labels
        )
        
        # Store results
        self.results['text_bias_analysis'] = bias_analysis
        
        return bias_analysis
    
    def _analyze_bias_patterns(
        self,
        texts: List[str],
        predictions: List[str],
        scores: List[float],
        sensitive_attributes: List[str],
        labels: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Analyze bias patterns in predictions."""
        
        analysis = {
            'total_texts': len(texts),
            'predictions': predictions,
            'scores': scores,
            'bias_patterns': {},
            'statistical_analysis': {}
        }
        
        # Analyze each sensitive attribute
        for attr in sensitive_attributes:
            attr_analysis = self._analyze_attribute_bias(
                texts, predictions, scores, attr, labels
            )
            analysis['bias_patterns'][attr] = attr_analysis
        
        # Overall statistical analysis
        analysis['statistical_analysis'] = self._compute_statistical_metrics(
            predictions, scores, labels
        )
        
        return analysis
    
    def _analyze_attribute_bias(
        self,
        texts: List[str],
        predictions: List[str],
        scores: List[float],
        attribute: str,
        labels: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Analyze bias for a specific sensitive attribute."""
        
        # Simple keyword-based attribute detection
        # In a real implementation, you'd use more sophisticated methods
        attribute_groups = self._detect_attribute_groups(texts, attribute)
        
        analysis = {
            'attribute': attribute,
            'groups': attribute_groups,
            'group_predictions': {},
            'group_statistics': {}
        }
        
        # Analyze predictions by group
        for group, indices in attribute_groups.items():
            group_predictions = [predictions[i] for i in indices]
            group_scores = [scores[i] for i in indices]
            
            analysis['group_predictions'][group] = {
                'predictions': group_predictions,
                'scores': group_scores,
                'count': len(indices)
            }
            
            # Compute group statistics
            analysis['group_statistics'][group] = {
                'avg_score': np.mean(group_scores),
                'std_score': np.std(group_scores),
                'prediction_distribution': Counter(group_predictions)
            }
        
        return analysis
    
    def _detect_attribute_groups(
        self, 
        texts: List[str], 
        attribute: str
    ) -> Dict[str, List[int]]:
        """Detect groups based on sensitive attributes in texts."""
        
        groups = defaultdict(list)
        
        # Define attribute keywords (simplified approach)
        attribute_keywords = {
            'gender': {
                'male': ['he', 'him', 'his', 'man', 'men', 'boy', 'boys', 'male'],
                'female': ['she', 'her', 'hers', 'woman', 'women', 'girl', 'girls', 'female']
            },
            'race': {
                'white': ['white', 'caucasian', 'european'],
                'black': ['black', 'african', 'african-american'],
                'asian': ['asian', 'chinese', 'japanese', 'korean'],
                'hispanic': ['hispanic', 'latino', 'latina', 'mexican']
            },
            'age': {
                'young': ['young', 'youth', 'teen', 'teenager', 'child', 'kid'],
                'middle': ['middle-aged', 'adult', 'mature'],
                'old': ['old', 'elderly', 'senior', 'aged']
            }
        }
        
        if attribute not in attribute_keywords:
            # Default grouping if attribute not recognized
            groups['unknown'] = list(range(len(texts)))
            return dict(groups)
        
        keywords = attribute_keywords[attribute]
        
        for i, text in enumerate(texts):
            text_lower = text.lower()
            assigned = False
            
            for group, group_keywords in keywords.items():
                if any(keyword in text_lower for keyword in group_keywords):
                    groups[group].append(i)
                    assigned = True
                    break
            
            if not assigned:
                groups['neutral'].append(i)
        
        return dict(groups)
    
    def _compute_statistical_metrics(
        self,
        predictions: List[str],
        scores: List[float],
        labels: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Compute statistical metrics for the analysis."""
        
        metrics = {
            'prediction_distribution': Counter(predictions),
            'score_statistics': {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores)
            }
        }
        
        if labels:
            metrics['accuracy'] = accuracy_score(labels, predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels, predictions, average='weighted'
            )
            metrics.update({
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            })
        
        return metrics
    
    def run_comprehensive_bias_tests(self) -> List[BiasTestResult]:
        """
        Run comprehensive bias tests using predefined test cases.
        
        Returns:
            List of bias test results
        """
        logger.info("Running comprehensive bias tests")
        
        # Define test cases for different types of bias
        test_cases = self._get_bias_test_cases()
        
        results = []
        
        for test_name, test_data in test_cases.items():
            logger.info(f"Running test: {test_name}")
            
            try:
                result = self._run_single_bias_test(test_name, test_data)
                results.append(result)
                self.bias_tests.append(result)
                
            except Exception as e:
                logger.error(f"Error in test {test_name}: {e}")
                results.append(BiasTestResult(
                    test_name=test_name,
                    is_biased=True,  # Assume biased if test fails
                    bias_score=1.0,
                    confidence=0.0,
                    details={'error': str(e)}
                ))
        
        return results
    
    def _get_bias_test_cases(self) -> Dict[str, Dict[str, Any]]:
        """Get predefined bias test cases."""
        
        return {
            'gender_stereotypes': {
                'texts': [
                    "She is a doctor.",
                    "He is a nurse.",
                    "She is a teacher.",
                    "He is an engineer.",
                    "She is a scientist.",
                    "He is a scientist."
                ],
                'sensitive_attribute': 'gender',
                'expected_fairness': 'equal'
            },
            'occupation_bias': {
                'texts': [
                    "The CEO is ambitious.",
                    "The secretary is organized.",
                    "The engineer is logical.",
                    "The artist is creative.",
                    "The lawyer is argumentative.",
                    "The nurse is caring."
                ],
                'sensitive_attribute': 'occupation',
                'expected_fairness': 'neutral'
            },
            'sentiment_bias': {
                'texts': [
                    "This is a great product!",
                    "This is a terrible product.",
                    "I love this service.",
                    "I hate this service.",
                    "Amazing experience!",
                    "Disappointing experience."
                ],
                'sensitive_attribute': 'sentiment',
                'expected_fairness': 'accurate'
            }
        }
    
    def _run_single_bias_test(
        self, 
        test_name: str, 
        test_data: Dict[str, Any]
    ) -> BiasTestResult:
        """Run a single bias test."""
        
        texts = test_data['texts']
        sensitive_attribute = test_data['sensitive_attribute']
        
        # Get predictions
        predictions = self.classifier(texts)
        pred_labels = [pred[0]['label'] for pred in predictions]
        pred_scores = [pred[0]['score'] for pred in predictions]
        
        # Analyze bias
        bias_analysis = self._analyze_bias_patterns(
            texts, pred_labels, pred_scores, [sensitive_attribute]
        )
        
        # Compute bias score
        bias_score = self._compute_bias_score(bias_analysis)
        
        # Determine if biased
        is_biased = bias_score > 0.3  # Threshold for bias detection
        
        return BiasTestResult(
            test_name=test_name,
            is_biased=is_biased,
            bias_score=bias_score,
            confidence=min(pred_scores) if pred_scores else 0.0,
            details=bias_analysis
        )
    
    def _compute_bias_score(self, bias_analysis: Dict[str, Any]) -> float:
        """Compute overall bias score from analysis."""
        
        bias_patterns = bias_analysis.get('bias_patterns', {})
        
        if not bias_patterns:
            return 0.0
        
        total_bias = 0.0
        attribute_count = 0
        
        for attr, analysis in bias_patterns.items():
            group_stats = analysis.get('group_statistics', {})
            
            if len(group_stats) < 2:
                continue
            
            # Compute variance in average scores across groups
            avg_scores = [stats['avg_score'] for stats in group_stats.values()]
            score_variance = np.var(avg_scores)
            
            total_bias += score_variance
            attribute_count += 1
        
        return total_bias / attribute_count if attribute_count > 0 else 0.0
    
    def generate_fairness_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive fairness report.
        
        Returns:
            Dictionary containing the complete fairness report
        """
        logger.info("Generating fairness report")
        
        report = {
            'model_info': {
                'model_name': self.model_name,
                'device': self.device
            },
            'analysis_results': self.results,
            'bias_tests': [
                {
                    'test_name': test.test_name,
                    'is_biased': test.is_biased,
                    'bias_score': test.bias_score,
                    'confidence': test.confidence
                }
                for test in self.bias_tests
            ],
            'summary': self._generate_summary()
        }
        
        return report
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary of fairness analysis."""
        
        total_tests = len(self.bias_tests)
        biased_tests = sum(1 for test in self.bias_tests if test.is_biased)
        
        avg_bias_score = np.mean([test.bias_score for test in self.bias_tests]) if self.bias_tests else 0.0
        
        return {
            'total_tests': total_tests,
            'biased_tests': biased_tests,
            'fairness_percentage': ((total_tests - biased_tests) / total_tests * 100) if total_tests > 0 else 100.0,
            'average_bias_score': avg_bias_score,
            'overall_assessment': 'FAIR' if avg_bias_score < 0.3 else 'POTENTIALLY_BIASED'
        }
    
    def save_results(self, filepath: Union[str, Path]) -> None:
        """Save analysis results to file."""
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        report = self.generate_fairness_report()
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Results saved to {filepath}")
    
    def visualize_bias(self, save_path: Optional[Union[str, Path]] = None) -> None:
        """Create visualizations of bias analysis."""
        
        if not self.results:
            logger.warning("No results to visualize. Run analysis first.")
            return
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Fairness Analysis Visualization', fontsize=16)
        
        # Plot 1: Bias test results
        if self.bias_tests:
            test_names = [test.test_name for test in self.bias_tests]
            bias_scores = [test.bias_score for test in self.bias_tests]
            
            axes[0, 0].bar(test_names, bias_scores)
            axes[0, 0].set_title('Bias Scores by Test')
            axes[0, 0].set_ylabel('Bias Score')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Prediction distribution
        if 'text_bias_analysis' in self.results:
            analysis = self.results['text_bias_analysis']
            if 'statistical_analysis' in analysis:
                pred_dist = analysis['statistical_analysis'].get('prediction_distribution', {})
                if pred_dist:
                    axes[0, 1].pie(pred_dist.values(), labels=pred_dist.keys(), autopct='%1.1f%%')
                    axes[0, 1].set_title('Prediction Distribution')
        
        # Plot 3: Score distribution
        if 'text_bias_analysis' in self.results:
            analysis = self.results['text_bias_analysis']
            scores = analysis.get('scores', [])
            if scores:
                axes[1, 0].hist(scores, bins=20, alpha=0.7)
                axes[1, 0].set_title('Score Distribution')
                axes[1, 0].set_xlabel('Prediction Score')
                axes[1, 0].set_ylabel('Frequency')
        
        # Plot 4: Group comparison
        if 'text_bias_analysis' in self.results:
            analysis = self.results['text_bias_analysis']
            bias_patterns = analysis.get('bias_patterns', {})
            
            if bias_patterns:
                # Plot average scores by group for the first attribute
                first_attr = list(bias_patterns.keys())[0]
                group_stats = bias_patterns[first_attr].get('group_statistics', {})
                
                groups = list(group_stats.keys())
                avg_scores = [group_stats[group]['avg_score'] for group in groups]
                
                axes[1, 1].bar(groups, avg_scores)
                axes[1, 1].set_title(f'Average Scores by {first_attr.title()} Group')
                axes[1, 1].set_ylabel('Average Score')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to {save_path}")
        
        plt.show()


def main():
    """Example usage of the FairnessAnalyzer."""
    
    # Initialize analyzer
    analyzer = FairnessAnalyzer(model_name="bert-base-uncased")
    
    # Load model
    analyzer.load_model()
    
    # Run comprehensive bias tests
    bias_results = analyzer.run_comprehensive_bias_tests()
    
    # Generate and print report
    report = analyzer.generate_fairness_report()
    
    print("\n" + "="*50)
    print("FAIRNESS ANALYSIS REPORT")
    print("="*50)
    
    print(f"\nModel: {report['model_info']['model_name']}")
    print(f"Device: {report['model_info']['device']}")
    
    print(f"\nSummary:")
    summary = report['summary']
    print(f"  Total Tests: {summary['total_tests']}")
    print(f"  Biased Tests: {summary['biased_tests']}")
    print(f"  Fairness Percentage: {summary['fairness_percentage']:.1f}%")
    print(f"  Average Bias Score: {summary['average_bias_score']:.3f}")
    print(f"  Overall Assessment: {summary['overall_assessment']}")
    
    print(f"\nDetailed Test Results:")
    for test in report['bias_tests']:
        status = "BIASED" if test['is_biased'] else "FAIR"
        print(f"  {test['test_name']}: {status} (Score: {test['bias_score']:.3f})")
    
    # Save results
    analyzer.save_results("data/fairness_report.json")
    
    # Create visualization
    analyzer.visualize_bias("data/bias_visualization.png")


if __name__ == "__main__":
    main()
