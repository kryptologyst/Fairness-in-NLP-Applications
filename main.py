"""
Main Entry Point for Fairness in NLP Applications

This module provides the main entry point for running fairness analysis
with various interfaces (CLI, web app, or programmatic).
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
import yaml
import json

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from fairness_analyzer import FairnessAnalyzer
from dataset_generator import SyntheticDatasetGenerator, DatasetConfig


def setup_logging(config: Dict[str, Any]) -> None:
    """Setup logging configuration."""
    
    log_config = config.get('logging', {})
    level = getattr(logging, log_config.get('level', 'INFO').upper())
    
    # Configure logging
    logging.basicConfig(
        level=level,
        format=log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_config.get('file', 'logs/fairness_analysis.log'))
        ]
    )


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    
    if config_path is None:
        config_path = Path(__file__).parent / "config" / "config.yaml"
    
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logging.warning(f"Config file not found at {config_path}, using defaults")
        return get_default_config()


def get_default_config() -> Dict[str, Any]:
    """Get default configuration."""
    
    return {
        'model': {
            'name': 'bert-base-uncased',
            'device': 'auto',
            'cache_dir': './models/cache'
        },
        'fairness': {
            'bias_threshold': 0.3,
            'sensitive_attributes': ['gender', 'race', 'age']
        },
        'dataset': {
            'synthetic_samples': 1000,
            'bias_level': 0.3
        },
        'output': {
            'results_dir': 'data/results',
            'reports_dir': 'data/reports'
        },
        'logging': {
            'level': 'INFO',
            'file': 'logs/fairness_analysis.log'
        }
    }


def run_cli_analysis(args) -> None:
    """Run fairness analysis from command line."""
    
    # Load configuration
    config = load_config(args.config)
    setup_logging(config)
    
    logging.info("Starting CLI fairness analysis")
    
    # Initialize analyzer
    model_config = config['model']
    analyzer = FairnessAnalyzer(
        model_name=model_config['name'],
        device=model_config.get('device'),
        cache_dir=model_config.get('cache_dir')
    )
    
    try:
        analyzer.load_model()
        logging.info("Model loaded successfully")
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        return
    
    # Generate synthetic dataset if requested
    if args.generate_dataset:
        logging.info("Generating synthetic dataset")
        
        dataset_config = DatasetConfig(
            num_samples=config['dataset']['synthetic_samples'],
            bias_level=config['dataset']['bias_level']
        )
        
        generator = SyntheticDatasetGenerator(dataset_config)
        df = generator.generate_dataset()
        
        # Save dataset
        output_dir = Path(config['output']['results_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        dataset_path = output_dir / "synthetic_dataset"
        generator.save_dataset(df, dataset_path)
        
        # Analyze the generated dataset
        texts = df['text'].tolist()
        sensitive_attrs = config['fairness']['sensitive_attributes']
        
        logging.info(f"Analyzing {len(texts)} texts for bias")
        results = analyzer.analyze_text_bias(texts, sensitive_attrs)
    
    # Run comprehensive bias tests
    logging.info("Running comprehensive bias tests")
    bias_tests = analyzer.run_comprehensive_bias_tests()
    
    # Generate report
    logging.info("Generating fairness report")
    report = analyzer.generate_fairness_report()
    
    # Save results
    output_dir = Path(config['output']['reports_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = output_dir / "fairness_report.json"
    analyzer.save_results(report_path)
    
    # Print summary
    print("\n" + "="*60)
    print("FAIRNESS ANALYSIS REPORT")
    print("="*60)
    
    print(f"\nModel: {report['model_info']['model_name']}")
    print(f"Device: {report['model_info']['device']}")
    
    summary = report['summary']
    print(f"\nSummary:")
    print(f"  Total Tests: {summary['total_tests']}")
    print(f"  Biased Tests: {summary['biased_tests']}")
    print(f"  Fairness Percentage: {summary['fairness_percentage']:.1f}%")
    print(f"  Average Bias Score: {summary['average_bias_score']:.3f}")
    print(f"  Overall Assessment: {summary['overall_assessment']}")
    
    print(f"\nDetailed Test Results:")
    for test in report['bias_tests']:
        status = "BIASED" if test['is_biased'] else "FAIR"
        print(f"  {test['test_name']}: {status} (Score: {test['bias_score']:.3f})")
    
    print(f"\nReport saved to: {report_path}")
    
    # Create visualization if requested
    if args.visualize:
        logging.info("Creating visualizations")
        viz_path = output_dir / "bias_visualization.png"
        analyzer.visualize_bias(viz_path)
        print(f"Visualization saved to: {viz_path}")


def run_web_app(args) -> None:
    """Run the Streamlit web application."""
    
    import subprocess
    import sys
    
    app_path = Path(__file__).parent / "web_app" / "app.py"
    
    cmd = [
        sys.executable, "-m", "streamlit", "run", 
        str(app_path),
        "--server.port", str(args.port),
        "--server.address", args.host
    ]
    
    if args.debug:
        cmd.extend(["--logger.level", "debug"])
    
    logging.info(f"Starting web app on {args.host}:{args.port}")
    subprocess.run(cmd)


def main():
    """Main entry point."""
    
    parser = argparse.ArgumentParser(
        description="Fairness Analysis for NLP Applications",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run CLI analysis with default settings
  python main.py analyze
  
  # Run CLI analysis with custom config
  python main.py analyze --config custom_config.yaml
  
  # Generate dataset and analyze
  python main.py analyze --generate-dataset --visualize
  
  # Run web application
  python main.py web --port 8501
  
  # Run web application with debug mode
  python main.py web --debug --host 0.0.0.0 --port 8080
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # CLI analysis command
    analyze_parser = subparsers.add_parser('analyze', help='Run fairness analysis')
    analyze_parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to configuration file'
    )
    analyze_parser.add_argument(
        '--generate-dataset', '-g',
        action='store_true',
        help='Generate synthetic dataset for analysis'
    )
    analyze_parser.add_argument(
        '--visualize', '-v',
        action='store_true',
        help='Create visualizations'
    )
    
    # Web app command
    web_parser = subparsers.add_parser('web', help='Run web application')
    web_parser.add_argument(
        '--host',
        type=str,
        default='localhost',
        help='Host to run the web app on'
    )
    web_parser.add_argument(
        '--port', '-p',
        type=int,
        default=8501,
        help='Port to run the web app on'
    )
    web_parser.add_argument(
        '--debug',
        action='store_true',
        help='Run in debug mode'
    )
    
    args = parser.parse_args()
    
    if args.command == 'analyze':
        run_cli_analysis(args)
    elif args.command == 'web':
        run_web_app(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
