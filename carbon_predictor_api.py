#!/usr/bin/env python3
"""
Carbon Emission Predictor API
=============================

A comprehensive interface for predicting carbon emissions from prompt characteristics
using the trained ML model.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class CarbonEmissionPredictor:
    """
    Production-ready carbon emission predictor with comprehensive feature engineering.
    """
    
    def __init__(self, model_path="best_carbon_predictor.pkl"):
        """Initialize the predictor by loading the trained model."""
        self.model_data = joblib.load(model_path)
        self.model = self.model_data['model']
        self.scaler = self.model_data['scaler']
        self.feature_columns = self.model_data['feature_columns']
        self.model_name = self.model_data['model_name']
        
        print(f"✅ Loaded {self.model_name} model")
        print(f"📊 Expected features: {len(self.feature_columns)}")
        
        # Initialize TF-IDF vectorizer (need to refit on original data for consistency)
        self.tfidf = TfidfVectorizer(
            max_features=100,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        
        # Load original data to fit TF-IDF
        self._fit_tfidf_from_original_data()
    
    def _fit_tfidf_from_original_data(self):
        """Fit TF-IDF vectorizer on original training data."""
        try:
            original_df = pd.read_csv("final_experiment_results.csv")
            self.tfidf.fit(original_df['Prompt_used'])
            print("📝 TF-IDF vectorizer fitted on original data")
        except FileNotFoundError:
            print("⚠️ Warning: Original data not found. TF-IDF may not work correctly.")
    
    def engineer_features(self, prompt_text, prompt_type, length_type, prompt_complexity, 
                         token_length, inference_time, energy, iteration_number=0):
        """
        Engineer features from input parameters to match the training data format.
        
        Args:
            prompt_text (str): The actual prompt text
            prompt_type (str): Type of prompt (zero_shot, few_shot, cot, alessio_user, detailed)
            length_type (str): Length category (short, medium)
            prompt_complexity (str): Complexity level (low, medium, high)
            token_length (int): Number of tokens in the prompt
            inference_time (float): Time taken for inference
            energy (float): Energy consumption
            iteration_number (int): Iteration number (default: 0)
        
        Returns:
            pd.DataFrame: Engineered features ready for prediction
        """
        
        # Basic input validation
        valid_prompt_types = ['zero_shot', 'few_shot', 'cot', 'alessio_user', 'detailed']
        valid_length_types = ['short', 'medium']
        valid_complexity_levels = ['low', 'medium', 'high']
        
        if prompt_type not in valid_prompt_types:
            raise ValueError(f"Invalid prompt_type. Must be one of: {valid_prompt_types}")
        if length_type not in valid_length_types:
            raise ValueError(f"Invalid length_type. Must be one of: {valid_length_types}")
        if prompt_complexity not in valid_complexity_levels:
            raise ValueError(f"Invalid prompt_complexity. Must be one of: {valid_complexity_levels}")
        
        # Create a dictionary to store features
        features = {}
        
        # Basic features
        features['Token_Length'] = token_length
        features['inference_time'] = inference_time
        features['energy'] = energy
        features['iteration_number'] = iteration_number
        
        # Text-based features
        words = prompt_text.split()
        features['prompt_word_count'] = len(words)
        features['prompt_char_count'] = len(prompt_text)
        features['prompt_avg_word_length'] = len(prompt_text) / len(words) if words else 0
        features['prompt_has_quotes'] = 1 if '"' in prompt_text else 0
        features['prompt_has_question'] = 1 if '?' in prompt_text else 0
        features['prompt_complexity_score'] = (
            len(words) * 0.3 + 
            len(prompt_text) * 0.001 + 
            token_length * 0.7
        )
        
        # Efficiency features
        features['energy_per_token'] = energy / (token_length + 1)
        features['energy_efficiency'] = token_length / (inference_time + 0.001)
        
        # Categorical mappings
        complexity_map = {'low': 1, 'medium': 2, 'high': 3}
        length_map = {'short': 1, 'medium': 2}
        
        features['complexity_numeric'] = complexity_map[prompt_complexity]
        features['length_numeric'] = length_map[length_type]
        features['complexity_length_interaction'] = (
            features['complexity_numeric'] * features['length_numeric']
        )
        
        # Response features (we'll use dummy values since we don't have actual response)
        # In a real scenario, these would be calculated from the actual model response
        features['response_length'] = token_length * 3  # Estimate
        features['response_word_count'] = token_length * 0.75  # Estimate
        
        # One-hot encode categorical features
        for ptype in ['zero_shot', 'few_shot', 'cot', 'alessio_user', 'detailed']:
            features[f'Prompt_type_{ptype}'] = 1 if prompt_type == ptype else 0
            
        for ltype in ['short', 'medium']:
            features[f'Length_type_{ltype}'] = 1 if length_type == ltype else 0
            
        for comp in ['low', 'medium', 'high']:
            features[f'Prompt_complexity_{comp}'] = 1 if prompt_complexity == comp else 0
        
        # TF-IDF features
        try:
            tfidf_vector = self.tfidf.transform([prompt_text]).toarray()[0]
            for i, tfidf_val in enumerate(tfidf_vector):
                features[f'tfidf_{i}'] = tfidf_val
        except Exception as e:
            print(f"⚠️ Warning: TF-IDF features not available: {e}")
            # Fill with zeros if TF-IDF fails
            for i in range(100):
                features[f'tfidf_{i}'] = 0.0
        
        # Convert to DataFrame
        feature_df = pd.DataFrame([features])
        
        # Ensure all expected columns are present
        for col in self.feature_columns:
            if col not in feature_df.columns:
                feature_df[col] = 0.0
        
        # Reorder columns to match training data
        feature_df = feature_df[self.feature_columns]
        
        return feature_df
    
    def predict_carbon_emission(self, prompt_text, prompt_type, length_type, prompt_complexity,
                               token_length, inference_time, energy, iteration_number=0):
        """
        Predict carbon emission for given prompt characteristics.
        
        Returns:
            dict: Prediction results with detailed breakdown
        """
        
        # Engineer features
        features = self.engineer_features(
            prompt_text, prompt_type, length_type, prompt_complexity,
            token_length, inference_time, energy, iteration_number
        )
        
        # Make prediction
        prediction = self.model.predict(features)[0]
        
        # Calculate some insights
        energy_per_token = energy / (token_length + 1)
        carbon_per_token = prediction / (token_length + 1)
        efficiency_score = token_length / (inference_time + 0.001)
        
        # Determine efficiency rating
        if efficiency_score > 10:
            efficiency_rating = "Excellent"
        elif efficiency_score > 5:
            efficiency_rating = "Good"
        elif efficiency_score > 2:
            efficiency_rating = "Average"
        else:
            efficiency_rating = "Poor"
        
        return {
            'predicted_carbon_emission': prediction,
            'energy_consumption': energy,
            'carbon_per_token': carbon_per_token,
            'energy_per_token': energy_per_token,
            'efficiency_score': efficiency_score,
            'efficiency_rating': efficiency_rating,
            'prompt_complexity_score': features['prompt_complexity_score'].iloc[0],
            'model_used': self.model_name,
            'confidence': 'High' if self.model_name == 'Linear Regression' else 'Medium'
        }
    
    def predict_from_existing_data(self, data_row):
        """
        Predict carbon emission from an existing data row (for validation).
        
        Args:
            data_row (dict): Row of data with all required fields
        
        Returns:
            dict: Prediction results
        """
        return self.predict_carbon_emission(
            prompt_text=data_row['Prompt_used'],
            prompt_type=data_row['Prompt_type'],
            length_type=data_row['Length_type'],
            prompt_complexity=data_row['Prompt_complexity'],
            token_length=data_row['Token_Length'],
            inference_time=data_row['inference_time'],
            energy=data_row['energy'],
            iteration_number=data_row.get('iteration_number', 0)
        )
    
    def batch_predict(self, data_list):
        """
        Predict carbon emissions for a batch of prompts.
        
        Args:
            data_list (list): List of dictionaries with prompt data
        
        Returns:
            list: List of prediction results
        """
        results = []
        for i, data in enumerate(data_list):
            try:
                result = self.predict_carbon_emission(**data)
                result['batch_index'] = i
                results.append(result)
            except Exception as e:
                print(f"⚠️ Error predicting for batch item {i}: {e}")
                results.append({'error': str(e), 'batch_index': i})
        
        return results
    
    def get_model_info(self):
        """Get information about the loaded model."""
        return {
            'model_name': self.model_name,
            'num_features': len(self.feature_columns),
            'feature_categories': {
                'numerical': len([col for col in self.feature_columns if not col.startswith(('Prompt_type', 'Length_type', 'Prompt_complexity', 'tfidf_'))]),
                'categorical': len([col for col in self.feature_columns if col.startswith(('Prompt_type', 'Length_type', 'Prompt_complexity'))]),
                'text_features': len([col for col in self.feature_columns if col.startswith('tfidf_')])
            }
        }

def demo_predictions():
    """Demonstrate the predictor with sample data."""
    print("🚀 CARBON EMISSION PREDICTOR DEMO")
    print("="*50)
    
    # Initialize predictor
    predictor = CarbonEmissionPredictor()
    
    print(f"\n📋 Model Info:")
    model_info = predictor.get_model_info()
    for key, value in model_info.items():
        print(f"   {key}: {value}")
    
    # Sample predictions
    sample_data = [
        {
            'prompt_text': 'Classify this requirement as functional or non-functional: "The system shall process 1000 transactions per second."',
            'prompt_type': 'zero_shot',
            'length_type': 'short',
            'prompt_complexity': 'low',
            'token_length': 25,
            'inference_time': 15.5,
            'energy': 0.095
        },
        {
            'prompt_text': 'Given the following examples of functional requirements: [...] Now classify this new requirement with detailed reasoning.',
            'prompt_type': 'few_shot',
            'length_type': 'medium',
            'prompt_complexity': 'high',
            'token_length': 85,
            'inference_time': 45.2,
            'energy': 0.285
        },
        {
            'prompt_text': 'Think step by step about whether this requirement is functional or non-functional, considering each aspect carefully.',
            'prompt_type': 'cot',
            'length_type': 'medium',
            'prompt_complexity': 'medium',
            'token_length': 35,
            'inference_time': 28.1,
            'energy': 0.175
        }
    ]
    
    print(f"\n🔮 SAMPLE PREDICTIONS:")
    print("="*50)
    
    for i, data in enumerate(sample_data, 1):
        print(f"\n📝 Sample {i}: {data['prompt_type']} - {data['prompt_complexity']} complexity")
        print(f"   Prompt: {data['prompt_text'][:80]}...")
        
        try:
            result = predictor.predict_carbon_emission(**data)
            
            print(f"   🌱 Predicted Carbon Emission: {result['predicted_carbon_emission']:.8f}")
            print(f"   ⚡ Energy Consumption: {result['energy_consumption']:.6f}")
            print(f"   📊 Efficiency Rating: {result['efficiency_rating']}")
            print(f"   💡 Carbon per Token: {result['carbon_per_token']:.8f}")
            print(f"   🎯 Confidence: {result['confidence']}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Validation with actual data
    print(f"\n✅ VALIDATION WITH ACTUAL DATA:")
    print("="*50)
    
    try:
        # Load a sample from the original dataset
        original_df = pd.read_csv("final_experiment_results.csv")
        sample_row = original_df.iloc[0].to_dict()
        
        prediction_result = predictor.predict_from_existing_data(sample_row)
        actual_carbon = sample_row['carbon_emission']
        predicted_carbon = prediction_result['predicted_carbon_emission']
        
        print(f"   📊 Actual Carbon Emission: {actual_carbon:.8f}")
        print(f"   🔮 Predicted Carbon Emission: {predicted_carbon:.8f}")
        print(f"   📏 Absolute Error: {abs(actual_carbon - predicted_carbon):.8f}")
        print(f"   📈 Relative Error: {abs(actual_carbon - predicted_carbon) / actual_carbon * 100:.4f}%")
        
    except Exception as e:
        print(f"   ❌ Validation error: {e}")

if __name__ == "__main__":
    demo_predictions()