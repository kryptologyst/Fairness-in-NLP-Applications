"""
Test Suite for Fairness in NLP Applications

This module contains comprehensive tests for the fairness analysis system.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import json
import sys

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from fairness_analyzer import FairnessAnalyzer, BiasTestResult, FairnessMetrics
from dataset_generator import SyntheticDatasetGenerator, DatasetConfig


class TestFairnessAnalyzer:
    """Test cases for FairnessAnalyzer class."""
    
    @pytest.fixture
    def analyzer(self):
        """Create a FairnessAnalyzer instance for testing."""
        return FairnessAnalyzer(model_name="bert-base-uncased")
    
    @pytest.fixture
    def mock_classifier(self):
        """Create a mock classifier for testing."""
        mock_classifier = Mock()
        mock_classifier.return_value = [
            [{'label': 'POSITIVE', 'score': 0.8}],
            [{'label': 'NEGATIVE', 'score': 0.7}]
        ]
        return mock_classifier
    
    def test_analyzer_initialization(self, analyzer):
        """Test analyzer initialization."""
        assert analyzer.model_name == "bert-base-uncased"
        assert analyzer.device in ["cpu", "cuda"]
        assert analyzer.classifier is None
        assert analyzer.results == {}
        assert analyzer.bias_tests == []
    
    def test_load_model_mock(self, analyzer):
        """Test model loading with mocked components."""
        with patch('transformers.pipeline') as mock_pipeline, \
             patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer, \
             patch('transformers.AutoModelForSequenceClassification.from_pretrained') as mock_model:
            
            mock_pipeline.return_value = Mock()
            mock_tokenizer.return_value = Mock()
            mock_model.return_value = Mock()
            
            analyzer.load_model()
            
            assert analyzer.classifier is not None
            assert analyzer.tokenizer is not None
            assert analyzer.model is not None
    
    def test_analyze_text_bias(self, analyzer, mock_classifier):
        """Test text bias analysis."""
        analyzer.classifier = mock_classifier
        
        texts = ["She is a doctor.", "He is a nurse."]
        sensitive_attributes = ["gender"]
        
        results = analyzer.analyze_text_bias(texts, sensitive_attributes)
        
        assert 'total_texts' in results
        assert 'predictions' in results
        assert 'scores' in results
        assert 'bias_patterns' in results
        assert results['total_texts'] == 2
    
    def test_detect_attribute_groups(self, analyzer):
        """Test attribute group detection."""
        texts = [
            "She is a doctor.",
            "He is a nurse.",
            "The person is skilled."
        ]
        
        groups = analyzer._detect_attribute_groups(texts, "gender")
        
        assert 'female' in groups
        assert 'male' in groups
        assert 'neutral' in groups
        assert len(groups['female']) > 0
        assert len(groups['male']) > 0
    
    def test_compute_bias_score(self, analyzer):
        """Test bias score computation."""
        bias_analysis = {
            'bias_patterns': {
                'gender': {
                    'group_statistics': {
                        'male': {'avg_score': 0.8},
                        'female': {'avg_score': 0.6}
                    }
                }
            }
        }
        
        bias_score = analyzer._compute_bias_score(bias_analysis)
        
        assert isinstance(bias_score, float)
        assert bias_score >= 0.0
    
    def test_run_comprehensive_bias_tests(self, analyzer, mock_classifier):
        """Test comprehensive bias tests."""
        analyzer.classifier = mock_classifier
        
        results = analyzer.run_comprehensive_bias_tests()
        
        assert isinstance(results, list)
        assert len(results) > 0
        
        for result in results:
            assert isinstance(result, BiasTestResult)
            assert hasattr(result, 'test_name')
            assert hasattr(result, 'is_biased')
            assert hasattr(result, 'bias_score')
    
    def test_generate_fairness_report(self, analyzer):
        """Test fairness report generation."""
        # Add some mock results
        analyzer.results = {'test': 'data'}
        analyzer.bias_tests = [
            BiasTestResult(
                test_name="test1",
                is_biased=False,
                bias_score=0.1,
                confidence=0.9,
                details={}
            )
        ]
        
        report = analyzer.generate_fairness_report()
        
        assert 'model_info' in report
        assert 'analysis_results' in report
        assert 'bias_tests' in report
        assert 'summary' in report
        
        summary = report['summary']
        assert 'total_tests' in summary
        assert 'biased_tests' in summary
        assert 'fairness_percentage' in summary
    
    def test_save_results(self, analyzer):
        """Test saving results to file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = Path(temp_dir) / "test_results.json"
            
            # Add some mock data
            analyzer.results = {'test': 'data'}
            analyzer.bias_tests = []
            
            analyzer.save_results(filepath)
            
            assert filepath.exists()
            
            with open(filepath, 'r') as f:
                data = json.load(f)
                assert 'model_info' in data


class TestSyntheticDatasetGenerator:
    """Test cases for SyntheticDatasetGenerator class."""
    
    @pytest.fixture
    def config(self):
        """Create a DatasetConfig for testing."""
        return DatasetConfig(
            num_samples=100,
            bias_level=0.3,
            noise_level=0.1
        )
    
    @pytest.fixture
    def generator(self, config):
        """Create a SyntheticDatasetGenerator instance for testing."""
        return SyntheticDatasetGenerator(config)
    
    def test_generator_initialization(self, generator, config):
        """Test generator initialization."""
        assert generator.config == config
        assert generator.random_seed == 42
        assert 'occupation_gender' in generator.templates
    
    def test_generate_dataset(self, generator):
        """Test dataset generation."""
        df = generator.generate_dataset()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert 'text' in df.columns
        assert 'scenario' in df.columns
        assert 'gender' in df.columns
        assert 'race' in df.columns
        assert 'age_group' in df.columns
    
    def test_generate_occupation_gender_sample(self, generator):
        """Test occupation-gender sample generation."""
        templates = generator.templates['occupation_gender']
        
        result = generator._generate_occupation_gender_sample(
            'female', 'white', 'middle-aged', templates
        )
        
        assert 'text' in result
        assert 'occupation' in result
        assert 'expected_bias' in result
        assert isinstance(result['text'], str)
        assert len(result['text']) > 0
    
    def test_generate_bias_analysis_dataset(self, generator):
        """Test bias analysis dataset generation."""
        df, metadata = generator.generate_bias_analysis_dataset()
        
        assert isinstance(df, pd.DataFrame)
        assert isinstance(metadata, dict)
        assert len(df) > 0
        assert 'scenario' in df.columns
        assert 'text' in df.columns
        assert 'total_scenarios' in metadata
    
    def test_save_dataset(self, generator):
        """Test dataset saving."""
        df = pd.DataFrame({
            'text': ['Sample text 1', 'Sample text 2'],
            'label': ['A', 'B']
        })
        
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = Path(temp_dir) / "test_dataset"
            
            generator.save_dataset(df, filepath)
            
            # Check CSV file
            csv_path = filepath.with_suffix('.csv')
            assert csv_path.exists()
            
            # Check JSON file
            json_path = filepath.with_suffix('.json')
            assert json_path.exists()
            
            # Verify content
            loaded_df = pd.read_csv(csv_path)
            assert len(loaded_df) == 2
            assert 'text' in loaded_df.columns


class TestBiasTestResult:
    """Test cases for BiasTestResult dataclass."""
    
    def test_bias_test_result_creation(self):
        """Test BiasTestResult creation."""
        result = BiasTestResult(
            test_name="test_gender_bias",
            is_biased=True,
            bias_score=0.5,
            confidence=0.8,
            details={'group1': 'data1'}
        )
        
        assert result.test_name == "test_gender_bias"
        assert result.is_biased is True
        assert result.bias_score == 0.5
        assert result.confidence == 0.8
        assert result.details == {'group1': 'data1'}


class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_analysis(self):
        """Test end-to-end analysis workflow."""
        # Create analyzer
        analyzer = FairnessAnalyzer(model_name="bert-base-uncased")
        
        # Mock the classifier to avoid loading actual model
        mock_classifier = Mock()
        mock_classifier.return_value = [
            [{'label': 'POSITIVE', 'score': 0.8}],
            [{'label': 'NEGATIVE', 'score': 0.7}],
            [{'label': 'POSITIVE', 'score': 0.9}],
            [{'label': 'NEGATIVE', 'score': 0.6}]
        ]
        analyzer.classifier = mock_classifier
        
        # Generate test data
        config = DatasetConfig(num_samples=4, bias_level=0.3)
        generator = SyntheticDatasetGenerator(config)
        df = generator.generate_dataset()
        
        # Run analysis
        texts = df['text'].tolist()
        results = analyzer.analyze_text_bias(texts, ['gender'])
        
        # Verify results
        assert results['total_texts'] == 4
        assert len(results['predictions']) == 4
        assert len(results['scores']) == 4
        
        # Run bias tests
        bias_tests = analyzer.run_comprehensive_bias_tests()
        assert len(bias_tests) > 0
        
        # Generate report
        report = analyzer.generate_fairness_report()
        assert 'summary' in report
        assert report['summary']['total_tests'] > 0


# Pytest configuration
@pytest.fixture(scope="session")
def setup_test_environment():
    """Setup test environment."""
    # Create temporary directories for test outputs
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup is handled by tempfile


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
