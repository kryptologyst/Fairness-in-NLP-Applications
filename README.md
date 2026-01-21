# Fairness in NLP Applications

A comprehensive toolkit for analyzing fairness and bias in Natural Language Processing (NLP) models using state-of-the-art techniques and modern Python best practices.

## Overview

This project provides tools to detect, analyze, and visualize bias in NLP models across various sensitive attributes including gender, race, age, and occupation. It includes both programmatic APIs and user-friendly web interfaces for comprehensive fairness analysis.

## Features

- **Comprehensive Bias Detection**: Analyze bias across multiple sensitive attributes
- **Multiple Analysis Methods**: Statistical parity, equalized odds, demographic parity
- **Synthetic Dataset Generation**: Create controlled datasets with known bias patterns
- **Interactive Web Interface**: User-friendly Streamlit application
- **Rich Visualizations**: Interactive plots and charts for bias analysis
- **Configurable Analysis**: YAML-based configuration system
- **Comprehensive Testing**: Full test suite with pytest
- **Detailed Reporting**: JSON reports with actionable insights

## Quick Start

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/kryptologyst/Fairness-in-NLP-Applications.git
   cd Fairness-in-NLP-Applications
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the web application**:
   ```bash
   python main.py web
   ```

4. **Open your browser** and navigate to `http://localhost:8501`

### Command Line Usage

```bash
# Run fairness analysis with synthetic dataset
python main.py analyze --generate-dataset --visualize

# Run analysis with custom configuration
python main.py analyze --config custom_config.yaml

# Run web application on custom port
python main.py web --port 8080 --host 0.0.0.0
```

## 📁 Project Structure

```
0557_Fairness_in_NLP_Applications/
├── src/                          # Source code
│   ├── fairness_analyzer.py     # Core fairness analysis module
│   └── dataset_generator.py     # Synthetic dataset generation
├── web_app/                      # Streamlit web application
│   └── app.py                   # Main web interface
├── tests/                        # Test suite
│   └── test_fairness.py         # Comprehensive tests
├── config/                       # Configuration files
│   └── config.yaml              # Main configuration
├── data/                         # Data storage
│   ├── results/                 # Analysis results
│   ├── reports/                 # Generated reports
│   └── visualizations/          # Charts and plots
├── models/                       # Model storage
│   ├── cache/                   # Hugging Face cache
│   └── saved/                   # Saved models
├── logs/                         # Log files
├── main.py                       # Main entry point
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore rules
└── README.md                     # This file
```

## 🔧 Configuration

The project uses YAML configuration files for easy customization. Key configuration options:

```yaml
# Model Configuration
model:
  name: "bert-base-uncased"        # Hugging Face model name
  device: "auto"                   # Device selection
  cache_dir: "./models/cache"      # Model cache directory

# Fairness Analysis
fairness:
  bias_threshold: 0.3              # Bias detection threshold
  sensitive_attributes:            # Attributes to analyze
    - gender
    - race
    - age
    - occupation

# Dataset Generation
dataset:
  synthetic_samples: 1000         # Number of synthetic samples
  bias_level: 0.3                 # Controlled bias level
  noise_level: 0.1                # Noise in data
```

## Usage Examples

### 1. Programmatic Analysis

```python
from src.fairness_analyzer import FairnessAnalyzer

# Initialize analyzer
analyzer = FairnessAnalyzer(model_name="bert-base-uncased")
analyzer.load_model()

# Analyze texts for bias
texts = [
    "She is a doctor.",
    "He is a nurse.",
    "The engineer is skilled."
]

results = analyzer.analyze_text_bias(texts, ["gender"])

# Run comprehensive bias tests
bias_tests = analyzer.run_comprehensive_bias_tests()

# Generate report
report = analyzer.generate_fairness_report()
print(f"Fairness percentage: {report['summary']['fairness_percentage']:.1f}%")
```

### 2. Synthetic Dataset Generation

```python
from src.dataset_generator import SyntheticDatasetGenerator, DatasetConfig

# Create configuration
config = DatasetConfig(
    num_samples=500,
    bias_level=0.3,
    noise_level=0.1
)

# Generate dataset
generator = SyntheticDatasetGenerator(config)
df = generator.generate_dataset()

# Save dataset
generator.save_dataset(df, "data/my_dataset")
```

### 3. Web Interface

The Streamlit web interface provides:

- **Interactive Model Selection**: Choose from different pre-trained models
- **Custom Text Analysis**: Upload your own texts for analysis
- **Synthetic Dataset Generation**: Generate controlled datasets with known bias
- **Real-time Visualizations**: Interactive charts and plots
- **Comprehensive Reports**: Download detailed analysis reports

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_fairness.py -v
```

## Key Metrics

The system analyzes several fairness metrics:

- **Demographic Parity**: Equal positive prediction rates across groups
- **Equalized Odds**: Equal true/false positive rates across groups
- **Equal Opportunity**: Equal true positive rates across groups
- **Statistical Parity**: Equal prediction distributions across groups

## Bias Detection Methods

### 1. Statistical Analysis
- Variance analysis across demographic groups
- Confidence score distributions
- Prediction pattern analysis

### 2. Controlled Testing
- Predefined bias test cases
- Synthetic scenarios with known bias
- Cross-validation across sensitive attributes

### 3. Visualization
- Interactive bias score charts
- Group comparison plots
- Score distribution histograms

## Web Interface Features

### Home Tab
- Introduction to fairness in NLP
- Usage instructions
- Key metrics explanation

### Analysis Tab
- **Synthetic Dataset**: Generate and analyze controlled datasets
- **Custom Text Input**: Analyze your own texts
- **File Upload**: Upload CSV/JSON files for analysis

### Visualizations Tab
- Interactive bias score charts
- Prediction distribution plots
- Group comparison visualizations

### Reports Tab
- Comprehensive fairness reports
- Downloadable JSON reports
- Summary statistics

## Advanced Configuration

### Custom Model Integration

```python
# Use custom Hugging Face model
analyzer = FairnessAnalyzer(
    model_name="roberta-base",
    device="cuda",
    cache_dir="/path/to/cache"
)
```

### Custom Bias Tests

```python
# Add custom bias test cases
custom_tests = {
    'my_custom_test': {
        'texts': ["Custom text 1", "Custom text 2"],
        'sensitive_attribute': 'custom_attr',
        'expected_fairness': 'neutral'
    }
}
```

## 🔧 Development

### Setting up Development Environment

```bash
# Install development dependencies
pip install -r requirements.txt

# Install pre-commit hooks
pre-commit install

# Run code formatting
black src/ tests/
flake8 src/ tests/

# Run type checking
mypy src/
```

### Adding New Features

1. **New Bias Detection Methods**: Add to `FairnessAnalyzer` class
2. **New Visualizations**: Extend the visualization methods
3. **New Test Cases**: Add to the bias test suite
4. **Web Interface Components**: Extend the Streamlit app

## API Reference

### FairnessAnalyzer

Main class for fairness analysis.

```python
class FairnessAnalyzer:
    def __init__(self, model_name: str, device: str = None)
    def load_model(self) -> None
    def analyze_text_bias(self, texts: List[str], sensitive_attributes: List[str]) -> Dict
    def run_comprehensive_bias_tests(self) -> List[BiasTestResult]
    def generate_fairness_report(self) -> Dict
    def visualize_bias(self, save_path: str = None) -> None
```

### SyntheticDatasetGenerator

Class for generating synthetic datasets.

```python
class SyntheticDatasetGenerator:
    def __init__(self, config: DatasetConfig)
    def generate_dataset(self) -> pd.DataFrame
    def generate_bias_analysis_dataset(self) -> Tuple[pd.DataFrame, Dict]
    def save_dataset(self, df: pd.DataFrame, filepath: str) -> None
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Hugging Face](https://huggingface.co/) for pre-trained models and transformers library
- [Streamlit](https://streamlit.io/) for the web interface framework
- [scikit-learn](https://scikit-learn.org/) for machine learning utilities
- [Plotly](https://plotly.com/) for interactive visualizations

## Future Enhancements

- [ ] Support for more model architectures
- [ ] Advanced bias mitigation techniques
- [ ] Real-time bias monitoring
- [ ] Integration with MLflow for experiment tracking
- [ ] Support for multilingual bias analysis
- [ ] API endpoints for production deployment
# Fairness-in-NLP-Applications
# Fairness-in-NLP-Applications
