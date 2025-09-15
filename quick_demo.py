#!/usr/bin/env python3
"""
Simple CLI script for quick fairness analysis

This script provides a simple command-line interface for running
fairness analysis without the full web application.
"""

import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from fairness_analyzer import FairnessAnalyzer
from dataset_generator import SyntheticDatasetGenerator, DatasetConfig


def quick_analysis():
    """Run a quick fairness analysis demo."""
    
    print("⚖️ Fairness in NLP Applications - Quick Demo")
    print("=" * 50)
    
    # Initialize analyzer
    print("🔧 Initializing analyzer...")
    analyzer = FairnessAnalyzer(model_name="bert-base-uncased")
    
    try:
        analyzer.load_model()
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("💡 Make sure you have internet connection for downloading the model")
        return
    
    # Generate synthetic dataset
    print("\n🧪 Generating synthetic dataset...")
    config = DatasetConfig(num_samples=50, bias_level=0.3)
    generator = SyntheticDatasetGenerator(config)
    df = generator.generate_dataset()
    
    print(f"✅ Generated {len(df)} samples")
    
    # Run analysis
    print("\n🔍 Running fairness analysis...")
    texts = df['text'].tolist()
    results = analyzer.analyze_text_bias(texts, ['gender', 'race'])
    
    print(f"✅ Analyzed {results['total_texts']} texts")
    
    # Run bias tests
    print("\n🧪 Running comprehensive bias tests...")
    bias_tests = analyzer.run_comprehensive_bias_tests()
    
    print(f"✅ Completed {len(bias_tests)} bias tests")
    
    # Generate report
    print("\n📊 Generating fairness report...")
    report = analyzer.generate_fairness_report()
    
    # Display summary
    print("\n" + "=" * 50)
    print("📋 FAIRNESS ANALYSIS SUMMARY")
    print("=" * 50)
    
    print(f"Model: {report['model_info']['model_name']}")
    print(f"Device: {report['model_info']['device']}")
    
    summary = report['summary']
    print(f"\nTotal Tests: {summary['total_tests']}")
    print(f"Biased Tests: {summary['biased_tests']}")
    print(f"Fairness Percentage: {summary['fairness_percentage']:.1f}%")
    print(f"Average Bias Score: {summary['average_bias_score']:.3f}")
    
    assessment = summary['overall_assessment']
    if assessment == 'FAIR':
        print("🎉 Overall Assessment: FAIR")
    else:
        print("⚠️ Overall Assessment: POTENTIALLY BIASED")
    
    print("\n📊 Detailed Test Results:")
    for test in report['bias_tests']:
        status = "🔴 BIASED" if test['is_biased'] else "🟢 FAIR"
        print(f"  {test['test_name']}: {status} (Score: {test['bias_score']:.3f})")
    
    # Save results
    print("\n💾 Saving results...")
    analyzer.save_results("data/quick_analysis_report.json")
    print("✅ Results saved to data/quick_analysis_report.json")
    
    print("\n🎉 Analysis completed successfully!")
    print("💡 Run 'python main.py web' to use the interactive web interface")


if __name__ == "__main__":
    quick_analysis()
