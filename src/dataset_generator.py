"""
Synthetic Dataset Generator for Fairness Testing

This module generates synthetic datasets for comprehensive fairness testing
in NLP applications, including various demographic groups and scenarios.
"""

import json
import random
from typing import List, Dict, Any, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict


@dataclass
class DatasetConfig:
    """Configuration for synthetic dataset generation."""
    
    num_samples: int = 1000
    num_groups_per_attribute: int = 3
    bias_level: float = 0.2  # 0.0 = no bias, 1.0 = maximum bias
    noise_level: float = 0.1
    include_neutral: bool = True


class SyntheticDatasetGenerator:
    """
    Generator for synthetic datasets with controlled bias patterns.
    
    This class creates datasets with known bias patterns to test
    fairness analysis methods.
    """
    
    def __init__(self, config: DatasetConfig):
        """
        Initialize the dataset generator.
        
        Args:
            config: Configuration for dataset generation
        """
        self.config = config
        self.random_seed = 42
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        
        # Define templates for different scenarios
        self.templates = self._initialize_templates()
        
    def _initialize_templates(self) -> Dict[str, List[str]]:
        """Initialize text templates for different scenarios."""
        
        return {
            'occupation_gender': [
                "{pronoun} is a {occupation}.",
                "The {occupation} is {pronoun_lower}.",
                "{pronoun} works as a {occupation}.",
                "I met a {occupation} named {name}.",
                "The {occupation} {pronoun_lower} is very skilled."
            ],
            'sentiment_demographic': [
                "{pronoun} is {adjective}.",
                "The {demographic} person is {adjective}.",
                "{pronoun} seems {adjective}.",
                "I think {pronoun} is {adjective}.",
                "The {demographic} individual appears {adjective}."
            ],
            'professional_evaluation': [
                "{pronoun} is an excellent {profession}.",
                "The {profession} {pronoun_lower} performs well.",
                "{pronoun} has great skills as a {profession}.",
                "I recommend {pronoun} for the {profession} position.",
                "The {profession} {pronoun_lower} is highly qualified."
            ],
            'social_interaction': [
                "{pronoun} is {social_trait}.",
                "The {demographic} person is {social_trait}.",
                "{pronoun} behaves {social_trait}.",
                "I find {pronoun} {social_trait}.",
                "The {demographic} individual is {social_trait}."
            ]
        }
    
    def generate_dataset(self) -> pd.DataFrame:
        """
        Generate a synthetic dataset with controlled bias patterns.
        
        Returns:
            DataFrame containing the synthetic dataset
        """
        print(f"Generating synthetic dataset with {self.config.num_samples} samples...")
        
        data = []
        
        # Generate samples for each scenario
        samples_per_scenario = self.config.num_samples // len(self.templates)
        
        for scenario_name, templates in self.templates.items():
            scenario_data = self._generate_scenario_data(
                scenario_name, templates, samples_per_scenario
            )
            data.extend(scenario_data)
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Shuffle the dataset
        df = df.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
        
        print(f"Generated dataset with {len(df)} samples")
        return df
    
    def _generate_scenario_data(
        self, 
        scenario_name: str, 
        templates: List[str], 
        num_samples: int
    ) -> List[Dict[str, Any]]:
        """Generate data for a specific scenario."""
        
        data = []
        
        # Define demographic groups and their characteristics
        demographics = self._get_demographic_groups()
        
        for i in range(num_samples):
            # Select demographic attributes
            gender = random.choice(demographics['gender'])
            race = random.choice(demographics['race'])
            age_group = random.choice(demographics['age'])
            
            # Generate text based on scenario
            if scenario_name == 'occupation_gender':
                text_data = self._generate_occupation_gender_sample(
                    gender, race, age_group, templates
                )
            elif scenario_name == 'sentiment_demographic':
                text_data = self._generate_sentiment_demographic_sample(
                    gender, race, age_group, templates
                )
            elif scenario_name == 'professional_evaluation':
                text_data = self._generate_professional_evaluation_sample(
                    gender, race, age_group, templates
                )
            elif scenario_name == 'social_interaction':
                text_data = self._generate_social_interaction_sample(
                    gender, race, age_group, templates
                )
            else:
                text_data = self._generate_generic_sample(
                    gender, race, age_group, templates
                )
            
            # Add metadata
            text_data.update({
                'scenario': scenario_name,
                'gender': gender,
                'race': race,
                'age_group': age_group,
                'sample_id': f"{scenario_name}_{i}"
            })
            
            data.append(text_data)
        
        return data
    
    def _get_demographic_groups(self) -> Dict[str, List[str]]:
        """Get demographic group definitions."""
        
        return {
            'gender': ['male', 'female', 'non-binary'],
            'race': ['white', 'black', 'asian', 'hispanic'],
            'age': ['young', 'middle-aged', 'elderly']
        }
    
    def _generate_occupation_gender_sample(
        self, 
        gender: str, 
        race: str, 
        age_group: str, 
        templates: List[str]
    ) -> Dict[str, Any]:
        """Generate occupation-gender bias sample."""
        
        # Define occupations with potential gender bias
        occupations = {
            'male': ['engineer', 'doctor', 'CEO', 'scientist', 'programmer'],
            'female': ['nurse', 'teacher', 'secretary', 'artist', 'designer'],
            'neutral': ['manager', 'analyst', 'consultant', 'researcher', 'specialist']
        }
        
        # Select occupation based on gender (with bias)
        if random.random() < self.config.bias_level:
            # Apply bias
            occupation = random.choice(occupations.get(gender, occupations['neutral']))
        else:
            # No bias
            occupation = random.choice(occupations['neutral'])
        
        # Generate pronouns
        pronouns = {
            'male': ('He', 'he'),
            'female': ('She', 'she'),
            'non-binary': ('They', 'they')
        }
        
        pronoun, pronoun_lower = pronouns.get(gender, ('They', 'they'))
        
        # Generate name
        names = {
            'male': ['John', 'Michael', 'David', 'James', 'Robert'],
            'female': ['Sarah', 'Emily', 'Jessica', 'Ashley', 'Amanda'],
            'non-binary': ['Alex', 'Jordan', 'Taylor', 'Casey', 'Riley']
        }
        
        name = random.choice(names.get(gender, names['non-binary']))
        
        # Generate text
        template = random.choice(templates)
        text = template.format(
            pronoun=pronoun,
            pronoun_lower=pronoun_lower,
            occupation=occupation,
            name=name
        )
        
        return {
            'text': text,
            'occupation': occupation,
            'expected_bias': 'gender_occupation',
            'bias_level': self.config.bias_level
        }
    
    def _generate_sentiment_demographic_sample(
        self, 
        gender: str, 
        race: str, 
        age_group: str, 
        templates: List[str]
    ) -> Dict[str, Any]:
        """Generate sentiment-demographic bias sample."""
        
        # Define adjectives with potential demographic bias
        positive_adjectives = ['kind', 'intelligent', 'hardworking', 'creative', 'reliable']
        negative_adjectives = ['lazy', 'aggressive', 'unreliable', 'difficult', 'stubborn']
        
        # Apply bias based on demographic groups
        if random.random() < self.config.bias_level:
            # Apply bias (simplified - in reality this would be more nuanced)
            if race in ['black', 'hispanic']:
                adjective = random.choice(negative_adjectives)
                sentiment = 'negative'
            else:
                adjective = random.choice(positive_adjectives)
                sentiment = 'positive'
        else:
            # No bias
            adjective = random.choice(positive_adjectives + negative_adjectives)
            sentiment = 'positive' if adjective in positive_adjectives else 'negative'
        
        # Generate pronouns
        pronouns = {
            'male': ('He', 'he'),
            'female': ('She', 'she'),
            'non-binary': ('They', 'they')
        }
        
        pronoun, pronoun_lower = pronouns.get(gender, ('They', 'they'))
        
        # Generate text
        template = random.choice(templates)
        text = template.format(
            pronoun=pronoun,
            pronoun_lower=pronoun_lower,
            adjective=adjective,
            demographic=race
        )
        
        return {
            'text': text,
            'sentiment': sentiment,
            'adjective': adjective,
            'expected_bias': 'demographic_sentiment',
            'bias_level': self.config.bias_level
        }
    
    def _generate_professional_evaluation_sample(
        self, 
        gender: str, 
        race: str, 
        age_group: str, 
        templates: List[str]
    ) -> Dict[str, Any]:
        """Generate professional evaluation bias sample."""
        
        professions = ['engineer', 'manager', 'teacher', 'doctor', 'lawyer', 'artist']
        profession = random.choice(professions)
        
        # Generate pronouns
        pronouns = {
            'male': ('He', 'he'),
            'female': ('She', 'she'),
            'non-binary': ('They', 'they')
        }
        
        pronoun, pronoun_lower = pronouns.get(gender, ('They', 'they'))
        
        # Generate text
        template = random.choice(templates)
        text = template.format(
            pronoun=pronoun,
            pronoun_lower=pronoun_lower,
            profession=profession
        )
        
        return {
            'text': text,
            'profession': profession,
            'expected_bias': 'professional_evaluation',
            'bias_level': self.config.bias_level
        }
    
    def _generate_social_interaction_sample(
        self, 
        gender: str, 
        race: str, 
        age_group: str, 
        templates: List[str]
    ) -> Dict[str, Any]:
        """Generate social interaction bias sample."""
        
        social_traits = ['friendly', 'outgoing', 'shy', 'confident', 'aggressive', 'calm']
        social_trait = random.choice(social_traits)
        
        # Generate pronouns
        pronouns = {
            'male': ('He', 'he'),
            'female': ('She', 'she'),
            'non-binary': ('They', 'they')
        }
        
        pronoun, pronoun_lower = pronouns.get(gender, ('They', 'they'))
        
        # Generate text
        template = random.choice(templates)
        text = template.format(
            pronoun=pronoun,
            pronoun_lower=pronoun_lower,
            social_trait=social_trait,
            demographic=race
        )
        
        return {
            'text': text,
            'social_trait': social_trait,
            'expected_bias': 'social_interaction',
            'bias_level': self.config.bias_level
        }
    
    def _generate_generic_sample(
        self, 
        gender: str, 
        race: str, 
        age_group: str, 
        templates: List[str]
    ) -> Dict[str, Any]:
        """Generate generic sample."""
        
        # Generate pronouns
        pronouns = {
            'male': ('He', 'he'),
            'female': ('She', 'she'),
            'non-binary': ('They', 'they')
        }
        
        pronoun, pronoun_lower = pronouns.get(gender, ('They', 'they'))
        
        # Generate text
        template = random.choice(templates)
        text = template.format(
            pronoun=pronoun,
            pronoun_lower=pronoun_lower,
            demographic=race
        )
        
        return {
            'text': text,
            'expected_bias': 'generic',
            'bias_level': self.config.bias_level
        }
    
    def save_dataset(self, df: pd.DataFrame, filepath: Union[str, Path]) -> None:
        """Save dataset to file."""
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as CSV
        df.to_csv(filepath.with_suffix('.csv'), index=False)
        
        # Save as JSON for easy loading
        df.to_json(filepath.with_suffix('.json'), orient='records', indent=2)
        
        print(f"Dataset saved to {filepath}")
    
    def generate_bias_analysis_dataset(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Generate a dataset specifically designed for bias analysis.
        
        Returns:
            Tuple of (dataset, metadata)
        """
        
        print("Generating bias analysis dataset...")
        
        # Create controlled bias scenarios
        bias_scenarios = [
            {
                'name': 'gender_occupation_bias',
                'texts': [
                    "She is a nurse.",
                    "He is a doctor.",
                    "She is a teacher.",
                    "He is an engineer.",
                    "She is a secretary.",
                    "He is a CEO."
                ],
                'sensitive_attributes': ['gender'],
                'expected_bias_type': 'occupation_gender'
            },
            {
                'name': 'race_sentiment_bias',
                'texts': [
                    "The white person is intelligent.",
                    "The black person is aggressive.",
                    "The asian person is hardworking.",
                    "The hispanic person is lazy."
                ],
                'sensitive_attributes': ['race'],
                'expected_bias_type': 'race_sentiment'
            },
            {
                'name': 'age_competence_bias',
                'texts': [
                    "The young person is inexperienced.",
                    "The middle-aged person is competent.",
                    "The elderly person is outdated."
                ],
                'sensitive_attributes': ['age'],
                'expected_bias_type': 'age_competence'
            }
        ]
        
        data = []
        metadata = {
            'total_scenarios': len(bias_scenarios),
            'scenarios': []
        }
        
        for scenario in bias_scenarios:
            scenario_data = []
            
            for i, text in enumerate(scenario['texts']):
                data_point = {
                    'text': text,
                    'scenario': scenario['name'],
                    'scenario_id': i,
                    'sensitive_attributes': scenario['sensitive_attributes'],
                    'expected_bias_type': scenario['expected_bias_type'],
                    'sample_id': f"{scenario['name']}_{i}"
                }
                data.append(data_point)
                scenario_data.append(data_point)
            
            metadata['scenarios'].append({
                'name': scenario['name'],
                'num_samples': len(scenario['texts']),
                'sensitive_attributes': scenario['sensitive_attributes'],
                'expected_bias_type': scenario['expected_bias_type']
            })
        
        df = pd.DataFrame(data)
        
        print(f"Generated bias analysis dataset with {len(df)} samples")
        return df, metadata


def main():
    """Example usage of the SyntheticDatasetGenerator."""
    
    # Create configuration
    config = DatasetConfig(
        num_samples=500,
        bias_level=0.3,
        noise_level=0.1
    )
    
    # Initialize generator
    generator = SyntheticDatasetGenerator(config)
    
    # Generate main dataset
    df = generator.generate_dataset()
    
    # Save dataset
    generator.save_dataset(df, "data/synthetic_dataset")
    
    # Generate bias analysis dataset
    bias_df, metadata = generator.generate_bias_analysis_dataset()
    
    # Save bias analysis dataset
    generator.save_dataset(bias_df, "data/bias_analysis_dataset")
    
    # Save metadata
    with open("data/bias_analysis_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print("\nDataset generation completed!")
    print(f"Main dataset: {len(df)} samples")
    print(f"Bias analysis dataset: {len(bias_df)} samples")
    
    # Display sample data
    print("\nSample from main dataset:")
    print(df.head())
    
    print("\nSample from bias analysis dataset:")
    print(bias_df.head())


if __name__ == "__main__":
    main()
