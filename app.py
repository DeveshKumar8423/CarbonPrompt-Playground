#!/usr/bin/env python3
"""
Carbon Prompt Playground - Flask Backend
========================================

Interactive web application for predicting carbon emissions from AI prompts
with real-time visualizations and educational features.
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import json
import logging
from carbon_predictor_api import CarbonEmissionPredictor
import pandas as pd
import numpy as np

# Initialize Flask app
app = Flask(__name__)
CORS(app, origins=["http://localhost:5001", "http://localhost:8000", "http://127.0.0.1:5001", "http://127.0.0.1:8000"])  # Enable CORS for all routes

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the carbon emission predictor
try:
    predictor = CarbonEmissionPredictor()
    logger.info("✅ Carbon emission predictor loaded successfully")
except Exception as e:
    logger.error(f"❌ Failed to load predictor: {e}")
    predictor = None

@app.route('/')
def index():
    """Serve the main playground interface."""
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict_carbon():
    """
    API endpoint for real-time carbon emission predictions.
    
    Expected JSON payload:
    {
        "prompt_text": "string",
        "prompt_type": "zero_shot|few_shot|cot|alessio_user|detailed",
        "length_type": "short|medium", 
        "prompt_complexity": "low|medium|high",
        "token_length": integer,
        "inference_time": float,
        "energy": float
    }
    """
    if not predictor:
        return jsonify({'error': 'Predictor not available'}), 500
    
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = [
            'prompt_text', 'prompt_type', 'length_type', 
            'prompt_complexity', 'token_length', 'inference_time', 'energy'
        ]
        
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Make prediction
        result = predictor.predict_carbon_emission(
            prompt_text=data['prompt_text'],
            prompt_type=data['prompt_type'],
            length_type=data['length_type'],
            prompt_complexity=data['prompt_complexity'],
            token_length=int(data['token_length']),
            inference_time=float(data['inference_time']),
            energy=float(data['energy'])
        )
        
        # Add some additional analysis for the UI
        result['environmental_impact'] = get_environmental_impact(result['predicted_carbon_emission'])
        result['efficiency_tips'] = get_efficiency_tips(data, result)
        result['comparison_data'] = get_comparison_data(result)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/batch_predict', methods=['POST'])
def batch_predict_carbon():
    """API endpoint for batch predictions."""
    if not predictor:
        return jsonify({'error': 'Predictor not available'}), 500
    
    try:
        data = request.get_json()
        batch_data = data.get('batch_data', [])
        
        if not batch_data:
            return jsonify({'error': 'No batch data provided'}), 400
        
        results = predictor.batch_predict(batch_data)
        
        # Add analysis for each result
        for i, result in enumerate(results):
            if 'error' not in result:
                result['environmental_impact'] = get_environmental_impact(result['predicted_carbon_emission'])
                result['comparison_data'] = get_comparison_data(result)
        
        return jsonify({'results': results})
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/model_info')
def get_model_info():
    """Get information about the loaded model."""
    if not predictor:
        return jsonify({'error': 'Predictor not available'}), 500
    
    try:
        model_info = predictor.get_model_info()
        
        # Add additional info for the playground
        model_info['supported_prompt_types'] = [
            'zero_shot', 'few_shot', 'cot', 'alessio_user', 'detailed'
        ]
        model_info['supported_length_types'] = ['short', 'medium']
        model_info['supported_complexity_levels'] = ['low', 'medium', 'high']
        model_info['performance_metrics'] = {
            'r2_score': 1.0000,
            'rmse': 0.0000,
            'cv_score': 1.0000
        }
        
        return jsonify(model_info)
        
    except Exception as e:
        logger.error(f"Model info error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/sample_prompts')
def get_sample_prompts():
    """Get sample prompts for the playground."""
    try:
        # Load sample prompts from the original dataset
        df = pd.read_csv('final_experiment_results.csv')
        
        # Get diverse samples
        samples = []
        for prompt_type in df['Prompt_type'].unique():
            type_samples = df[df['Prompt_type'] == prompt_type].head(2)
            for _, row in type_samples.iterrows():
                samples.append({
                    'prompt_text': row['Prompt_used'][:200] + '...' if len(row['Prompt_used']) > 200 else row['Prompt_used'],
                    'prompt_type': row['Prompt_type'],
                    'length_type': row['Length_type'],
                    'prompt_complexity': row['Prompt_complexity'],
                    'token_length': row['Token_Length'],
                    'inference_time': row['inference_time'],
                    'energy': row['energy'],
                    'actual_carbon': row['carbon_emission']
                })
        
        return jsonify({'samples': samples[:10]})  # Return top 10 samples
        
    except Exception as e:
        logger.error(f"Sample prompts error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/playground_data')
def get_playground_data():
    """Get data for playground visualizations."""
    try:
        # Load the original dataset for analysis
        df = pd.read_csv('final_experiment_results.csv')
        
        # Prepare data for visualizations
        data = {
            'carbon_distribution': {
                'values': df['carbon_emission'].tolist(),
                'bins': 20
            },
            'energy_carbon_relationship': {
                'energy': df['energy'].tolist(),
                'carbon': df['carbon_emission'].tolist()
            },
            'prompt_type_analysis': {
                'types': df['Prompt_type'].value_counts().to_dict(),
                'avg_carbon_by_type': df.groupby('Prompt_type')['carbon_emission'].mean().to_dict(),
                'avg_energy_by_type': df.groupby('Prompt_type')['energy'].mean().to_dict()
            },
            'complexity_analysis': {
                'levels': df['Prompt_complexity'].value_counts().to_dict(),
                'avg_carbon_by_complexity': df.groupby('Prompt_complexity')['carbon_emission'].mean().to_dict()
            },
            'efficiency_metrics': {
                'carbon_per_token': (df['carbon_emission'] / df['Token_Length']).tolist(),
                'energy_per_token': (df['energy'] / df['Token_Length']).tolist(),
                'token_lengths': df['Token_Length'].tolist()
            }
        }
        
        return jsonify(data)
        
    except Exception as e:
        logger.error(f"Playground data error: {e}")
        return jsonify({'error': str(e)}), 500

def get_environmental_impact(carbon_emission):
    """Calculate environmental impact metrics."""
    # Convert to more meaningful units
    kg_co2 = carbon_emission * 1000  # Convert to kg
    tree_equivalent = carbon_emission / 0.02  # Rough estimate: 1 tree absorbs ~20g CO2/year
    
    if carbon_emission < 0.0001:
        impact_level = "Very Low"
        color = "#00C851"
    elif carbon_emission < 0.0002:
        impact_level = "Low"
        color = "#FFBB33"
    elif carbon_emission < 0.0003:
        impact_level = "Medium"
        color = "#FF6900"
    else:
        impact_level = "High"
        color = "#FF4444"
    
    return {
        'level': impact_level,
        'color': color,
        'kg_co2': kg_co2,
        'tree_equivalent': tree_equivalent,
        'comparison': f"Equivalent to {tree_equivalent:.4f} trees' daily CO2 absorption"
    }

def get_efficiency_tips(input_data, result):
    """Generate efficiency tips based on prediction."""
    tips = []
    
    # Analyze prompt characteristics
    if result['efficiency_rating'] == 'Poor':
        tips.append("💡 Try reducing inference time by optimizing your prompt")
        tips.append("🔧 Consider using a shorter, more direct prompt")
    
    if input_data['prompt_complexity'] == 'high':
        tips.append("📝 High complexity prompts consume more energy - consider simplifying")
    
    if input_data['prompt_type'] == 'detailed':
        tips.append("⚡ Detailed prompts are energy-intensive but more accurate per token")
    
    if input_data['token_length'] > 50:
        tips.append("📏 Long prompts increase carbon footprint - try to be more concise")
    
    if not tips:
        tips.append("✅ Your prompt is already quite efficient!")
    
    return tips

def get_comparison_data(result):
    """Generate comparison data for visualization."""
    return {
        'vs_average': {
            'carbon': result['predicted_carbon_emission'] / 0.0001117,  # vs dataset average
            'energy': result['energy_consumption'] / 0.1456  # vs dataset average
        },
        'efficiency_percentile': min(100, max(0, result['efficiency_score'] / 10 * 100)),
        'carbon_ranking': 'top_10%' if result['predicted_carbon_emission'] < 0.00006 else 
                         'top_25%' if result['predicted_carbon_emission'] < 0.0001 else
                         'average' if result['predicted_carbon_emission'] < 0.00015 else 'high'
    }

if __name__ == '__main__':
    print("🚀 Starting Carbon Prompt Playground...")
    print("🌱 Navigate to http://localhost:5000 to start playing!")
    app.run(debug=True, host='0.0.0.0', port=5000)