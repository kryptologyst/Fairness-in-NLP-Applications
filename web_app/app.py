"""
Streamlit Web Interface for Fairness in NLP Applications

This module provides a user-friendly web interface for running
fairness analysis on NLP models.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import yaml
from pathlib import Path
import sys
import logging
from typing import Dict, List, Any, Optional

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from fairness_analyzer import FairnessAnalyzer, BiasTestResult
from dataset_generator import SyntheticDatasetGenerator, DatasetConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Fairness in NLP Applications",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .bias-indicator {
        font-size: 1.2rem;
        font-weight: bold;
        padding: 0.5rem;
        border-radius: 0.3rem;
        text-align: center;
    }
    .fair {
        background-color: #d4edda;
        color: #155724;
    }
    .biased {
        background-color: #f8d7da;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)


class FairnessWebApp:
    """Streamlit web application for fairness analysis."""
    
    def __init__(self):
        """Initialize the web application."""
        self.analyzer: Optional[FairnessAnalyzer] = None
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            config_path = Path(__file__).parent.parent / "config" / "config.yaml"
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning("Config file not found, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'model': {'name': 'bert-base-uncased', 'device': 'auto'},
            'fairness': {'bias_threshold': 0.3},
            'visualization': {'save_plots': True}
        }
    
    def run(self):
        """Run the Streamlit application."""
        
        # Header
        st.markdown('<h1 class="main-header">⚖️ Fairness in NLP Applications</h1>', 
                   unsafe_allow_html=True)
        
        st.markdown("""
        This application analyzes fairness and bias in NLP models using state-of-the-art 
        techniques. Upload your own texts or use our synthetic dataset to test for 
        various types of bias including gender, race, age, and occupational stereotypes.
        """)
        
        # Sidebar
        self._render_sidebar()
        
        # Main content
        tab1, tab2, tab3, tab4 = st.tabs([
            "🏠 Home", 
            "📊 Analysis", 
            "📈 Visualizations", 
            "📋 Reports"
        ])
        
        with tab1:
            self._render_home_tab()
        
        with tab2:
            self._render_analysis_tab()
        
        with tab3:
            self._render_visualizations_tab()
        
        with tab4:
            self._render_reports_tab()
    
    def _render_sidebar(self):
        """Render the sidebar with configuration options."""
        
        st.sidebar.header("⚙️ Configuration")
        
        # Model selection
        model_name = st.sidebar.selectbox(
            "Model",
            ["bert-base-uncased", "roberta-base", "distilbert-base-uncased"],
            index=0
        )
        
        # Device selection
        device = st.sidebar.selectbox(
            "Device",
            ["auto", "cpu", "cuda"],
            index=0
        )
        
        # Bias threshold
        bias_threshold = st.sidebar.slider(
            "Bias Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.05,
            help="Threshold for determining if a model is biased"
        )
        
        # Store configuration in session state
        st.session_state.config = {
            'model_name': model_name,
            'device': device,
            'bias_threshold': bias_threshold
        }
        
        # Initialize analyzer button
        if st.sidebar.button("🚀 Initialize Analyzer", type="primary"):
            with st.spinner("Loading model..."):
                try:
                    self.analyzer = FairnessAnalyzer(
                        model_name=model_name,
                        device=device if device != "auto" else None
                    )
                    self.analyzer.load_model()
                    st.session_state.analyzer_ready = True
                    st.sidebar.success("✅ Analyzer ready!")
                except Exception as e:
                    st.sidebar.error(f"❌ Error: {str(e)}")
                    st.session_state.analyzer_ready = False
    
    def _render_home_tab(self):
        """Render the home tab."""
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.header("🎯 What is Fairness in NLP?")
            
            st.markdown("""
            Fairness in Natural Language Processing (NLP) refers to the principle that 
            NLP models should not exhibit biased behavior or make discriminatory decisions 
            based on sensitive attributes such as:
            
            - **Gender**: Male, female, non-binary
            - **Race/Ethnicity**: White, Black, Asian, Hispanic, etc.
            - **Age**: Young, middle-aged, elderly
            - **Occupation**: Various professional roles
            - **Other Protected Attributes**: Religion, sexual orientation, etc.
            
            ### Why is this important?
            
            Biased NLP models can perpetuate harmful stereotypes and lead to unfair 
            outcomes in real-world applications such as:
            
            - Hiring and recruitment systems
            - Credit scoring and loan approval
            - Content moderation
            - Sentiment analysis
            - Machine translation
            """)
        
        with col2:
            st.header("🔍 How to Use This Tool")
            
            steps = [
                "1. **Configure** your model and settings in the sidebar",
                "2. **Initialize** the analyzer by clicking the button",
                "3. **Upload** your own texts or use synthetic data",
                "4. **Run** fairness analysis",
                "5. **Review** results and visualizations",
                "6. **Export** reports for further analysis"
            ]
            
            for step in steps:
                st.markdown(step)
            
            st.header("📊 Key Metrics")
            
            metrics = [
                ("Demographic Parity", "Equal positive prediction rates across groups"),
                ("Equalized Odds", "Equal true/false positive rates"),
                ("Equal Opportunity", "Equal true positive rates"),
                ("Statistical Parity", "Equal prediction distributions")
            ]
            
            for metric, description in metrics:
                with st.expander(metric):
                    st.write(description)
    
    def _render_analysis_tab(self):
        """Render the analysis tab."""
        
        if not st.session_state.get('analyzer_ready', False):
            st.warning("⚠️ Please initialize the analyzer first using the sidebar.")
            return
        
        st.header("📊 Fairness Analysis")
        
        # Analysis options
        analysis_type = st.radio(
            "Choose analysis type:",
            ["Synthetic Dataset", "Custom Text Input", "File Upload"],
            horizontal=True
        )
        
        if analysis_type == "Synthetic Dataset":
            self._render_synthetic_analysis()
        elif analysis_type == "Custom Text Input":
            self._render_custom_text_analysis()
        elif analysis_type == "File Upload":
            self._render_file_upload_analysis()
    
    def _render_synthetic_analysis(self):
        """Render synthetic dataset analysis."""
        
        st.subheader("🧪 Synthetic Dataset Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            num_samples = st.slider("Number of samples", 50, 1000, 200)
            bias_level = st.slider("Bias level", 0.0, 1.0, 0.3, 0.1)
        
        with col2:
            include_gender = st.checkbox("Include gender bias", True)
            include_race = st.checkbox("Include race bias", True)
            include_age = st.checkbox("Include age bias", True)
        
        if st.button("🚀 Generate and Analyze Dataset", type="primary"):
            with st.spinner("Generating synthetic dataset..."):
                try:
                    # Generate dataset
                    config = DatasetConfig(
                        num_samples=num_samples,
                        bias_level=bias_level
                    )
                    generator = SyntheticDatasetGenerator(config)
                    df = generator.generate_dataset()
                    
                    # Store in session state
                    st.session_state.synthetic_df = df
                    
                    # Run analysis
                    with st.spinner("Running fairness analysis..."):
                        texts = df['text'].tolist()
                        sensitive_attrs = []
                        if include_gender:
                            sensitive_attrs.append('gender')
                        if include_race:
                            sensitive_attrs.append('race')
                        if include_age:
                            sensitive_attrs.append('age')
                        
                        results = self.analyzer.analyze_text_bias(
                            texts, sensitive_attrs
                        )
                        
                        # Run bias tests
                        bias_tests = self.analyzer.run_comprehensive_bias_tests()
                        
                        st.session_state.analysis_results = results
                        st.session_state.bias_tests = bias_tests
                        
                        st.success("✅ Analysis completed!")
                        
                        # Display results
                        self._display_analysis_results()
                        
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    def _render_custom_text_analysis(self):
        """Render custom text input analysis."""
        
        st.subheader("✍️ Custom Text Analysis")
        
        # Text input
        texts = st.text_area(
            "Enter texts to analyze (one per line):",
            height=200,
            placeholder="She is a doctor.\nHe is a nurse.\nThe engineer is skilled."
        )
        
        # Sensitive attributes
        sensitive_attrs = st.multiselect(
            "Select sensitive attributes to analyze:",
            ["gender", "race", "age", "occupation"],
            default=["gender"]
        )
        
        if st.button("🔍 Analyze Custom Texts", type="primary"):
            if not texts.strip():
                st.warning("Please enter some texts to analyze.")
                return
            
            text_list = [text.strip() for text in texts.split('\n') if text.strip()]
            
            if not sensitive_attrs:
                st.warning("Please select at least one sensitive attribute.")
                return
            
            with st.spinner("Analyzing texts..."):
                try:
                    results = self.analyzer.analyze_text_bias(
                        text_list, sensitive_attrs
                    )
                    
                    st.session_state.analysis_results = results
                    st.success("✅ Analysis completed!")
                    
                    # Display results
                    self._display_analysis_results()
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    def _render_file_upload_analysis(self):
        """Render file upload analysis."""
        
        st.subheader("📁 File Upload Analysis")
        
        uploaded_file = st.file_uploader(
            "Upload a CSV or JSON file:",
            type=['csv', 'json'],
            help="File should contain a 'text' column and optional metadata columns"
        )
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_json(uploaded_file)
                
                st.success(f"✅ File loaded: {len(df)} rows")
                
                # Display preview
                st.subheader("📋 Data Preview")
                st.dataframe(df.head())
                
                # Column selection
                text_col = st.selectbox("Select text column:", df.columns)
                
                # Sensitive attributes
                available_cols = [col for col in df.columns if col != text_col]
                sensitive_attrs = st.multiselect(
                    "Select sensitive attribute columns:",
                    available_cols
                )
                
                if st.button("🔍 Analyze Uploaded Data", type="primary"):
                    if not sensitive_attrs:
                        st.warning("Please select at least one sensitive attribute column.")
                        return
                    
                    with st.spinner("Analyzing uploaded data..."):
                        try:
                            texts = df[text_col].tolist()
                            results = self.analyzer.analyze_text_bias(
                                texts, sensitive_attrs
                            )
                            
                            st.session_state.analysis_results = results
                            st.success("✅ Analysis completed!")
                            
                            # Display results
                            self._display_analysis_results()
                            
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
                            
            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")
    
    def _display_analysis_results(self):
        """Display analysis results."""
        
        if 'analysis_results' not in st.session_state:
            return
        
        results = st.session_state['analysis_results']
        
        st.subheader("📊 Analysis Results")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Texts", results['total_texts'])
        
        with col2:
            st.metric("Predictions", len(set(results['predictions'])))
        
        with col3:
            avg_score = np.mean(results['scores'])
            st.metric("Avg Confidence", f"{avg_score:.3f}")
        
        with col4:
            if 'statistical_analysis' in results:
                stats = results['statistical_analysis']
                if 'accuracy' in stats:
                    st.metric("Accuracy", f"{stats['accuracy']:.3f}")
                else:
                    st.metric("Score Std", f"{stats['score_statistics']['std']:.3f}")
        
        # Bias patterns
        if 'bias_patterns' in results:
            st.subheader("🔍 Bias Patterns")
            
            for attr, analysis in results['bias_patterns'].items():
                with st.expander(f"Analysis for {attr.title()}"):
                    
                    if 'group_statistics' in analysis:
                        groups = list(analysis['group_statistics'].keys())
                        
                        if len(groups) > 1:
                            # Create comparison chart
                            group_data = []
                            for group in groups:
                                stats = analysis['group_statistics'][group]
                                group_data.append({
                                    'Group': group,
                                    'Avg Score': stats['avg_score'],
                                    'Count': stats['count']
                                })
                            
                            df_groups = pd.DataFrame(group_data)
                            
                            # Display metrics
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                fig = px.bar(
                                    df_groups, 
                                    x='Group', 
                                    y='Avg Score',
                                    title=f"Average Scores by {attr.title()} Group"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with col2:
                                fig = px.pie(
                                    df_groups,
                                    values='Count',
                                    names='Group',
                                    title=f"Sample Distribution by {attr.title()}"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                        
                        # Detailed statistics
                        st.write("**Detailed Statistics:**")
                        for group, stats in analysis['group_statistics'].items():
                            st.write(f"- **{group}**: {stats['count']} samples, "
                                   f"avg score: {stats['avg_score']:.3f}")
    
    def _render_visualizations_tab(self):
        """Render visualizations tab."""
        
        if 'analysis_results' not in st.session_state:
            st.warning("⚠️ Please run an analysis first.")
            return
        
        st.header("📈 Visualizations")
        
        results = st.session_state['analysis_results']
        
        # Create visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Prediction distribution
            pred_dist = Counter(results['predictions'])
            fig = px.pie(
                values=list(pred_dist.values()),
                names=list(pred_dist.keys()),
                title="Prediction Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Score distribution
            fig = px.histogram(
                x=results['scores'],
                title="Score Distribution",
                nbins=20
            )
            fig.update_xaxis(title="Prediction Score")
            fig.update_yaxis(title="Frequency")
            st.plotly_chart(fig, use_container_width=True)
        
        # Bias test results
        if 'bias_tests' in st.session_state:
            st.subheader("🧪 Bias Test Results")
            
            bias_tests = st.session_state['bias_tests']
            
            # Create bias test visualization
            test_names = [test.test_name for test in bias_tests]
            bias_scores = [test.bias_score for test in bias_tests]
            is_biased = [test.is_biased for test in bias_tests]
            
            df_tests = pd.DataFrame({
                'Test': test_names,
                'Bias Score': bias_scores,
                'Is Biased': is_biased
            })
            
            # Color bars based on bias
            colors = ['red' if biased else 'green' for biased in is_biased]
            
            fig = px.bar(
                df_tests,
                x='Test',
                y='Bias Score',
                color='Is Biased',
                color_discrete_map={True: 'red', False: 'green'},
                title="Bias Scores by Test"
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary table
            st.subheader("📋 Test Summary")
            
            summary_data = []
            for test in bias_tests:
                summary_data.append({
                    'Test Name': test.test_name,
                    'Bias Score': f"{test.bias_score:.3f}",
                    'Status': '🔴 Biased' if test.is_biased else '🟢 Fair',
                    'Confidence': f"{test.confidence:.3f}"
                })
            
            df_summary = pd.DataFrame(summary_data)
            st.dataframe(df_summary, use_container_width=True)
    
    def _render_reports_tab(self):
        """Render reports tab."""
        
        if 'analysis_results' not in st.session_state:
            st.warning("⚠️ Please run an analysis first.")
            return
        
        st.header("📋 Reports")
        
        # Generate comprehensive report
        if st.button("📊 Generate Comprehensive Report", type="primary"):
            with st.spinner("Generating report..."):
                try:
                    report = self.analyzer.generate_fairness_report()
                    
                    st.subheader("📄 Fairness Report")
                    
                    # Model info
                    st.write("**Model Information:**")
                    st.write(f"- Model: {report['model_info']['model_name']}")
                    st.write(f"- Device: {report['model_info']['device']}")
                    
                    # Summary
                    st.write("**Summary:**")
                    summary = report['summary']
                    st.write(f"- Total Tests: {summary['total_tests']}")
                    st.write(f"- Biased Tests: {summary['biased_tests']}")
                    st.write(f"- Fairness Percentage: {summary['fairness_percentage']:.1f}%")
                    st.write(f"- Average Bias Score: {summary['average_bias_score']:.3f}")
                    
                    # Overall assessment
                    assessment = summary['overall_assessment']
                    if assessment == 'FAIR':
                        st.markdown('<div class="bias-indicator fair">✅ FAIR</div>', 
                                  unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="bias-indicator biased">⚠️ POTENTIALLY BIASED</div>', 
                                  unsafe_allow_html=True)
                    
                    # Detailed results
                    st.subheader("🔍 Detailed Test Results")
                    
                    for test in report['bias_tests']:
                        with st.expander(f"{test['test_name']} - {'🔴 Biased' if test['is_biased'] else '🟢 Fair'}"):
                            st.write(f"**Bias Score:** {test['bias_score']:.3f}")
                            st.write(f"**Confidence:** {test['confidence']:.3f}")
                            st.write(f"**Status:** {'Biased' if test['is_biased'] else 'Fair'}")
                    
                    # Download report
                    report_json = json.dumps(report, indent=2, default=str)
                    st.download_button(
                        label="📥 Download Report (JSON)",
                        data=report_json,
                        file_name="fairness_report.json",
                        mime="application/json"
                    )
                    
                except Exception as e:
                    st.error(f"❌ Error generating report: {str(e)}")


def main():
    """Main function to run the Streamlit app."""
    app = FairnessWebApp()
    app.run()


if __name__ == "__main__":
    main()
