#!/usr/bin/env python3
"""
Usage Example for Carbon Emission Predictor
==========================================

This script demonstrates how to use the CarbonEmissionPredictor
for practical carbon emission predictions from prompt characteristics.
"""

from carbon_predictor_api import CarbonEmissionPredictor

def main():
    """Main function demonstrating various usage scenarios."""
    
    print("🌱 CARBON EMISSION PREDICTOR - USAGE EXAMPLES")
    print("="*60)
    
    # Initialize the predictor
    predictor = CarbonEmissionPredictor()
    
    print("\n📚 SCENARIO 1: Single Prediction")
    print("-" * 40)
    
    # Example 1: Predict for a simple classification prompt
    result = predictor.predict_carbon_emission(
        prompt_text="Is this a functional requirement: 'The system must be user-friendly'",
        prompt_type="zero_shot",
        length_type="short", 
        prompt_complexity="low",
        token_length=15,
        inference_time=12.3,
        energy=0.078
    )
    
    print(f"Predicted Carbon Emission: {result['predicted_carbon_emission']:.8f}")
    print(f"Energy Efficiency Rating: {result['efficiency_rating']}")
    print(f"Carbon per Token: {result['carbon_per_token']:.8f}")
    
    print("\n📊 SCENARIO 2: Batch Predictions")
    print("-" * 40)
    
    # Example 2: Batch predictions for multiple prompts
    batch_data = [
        {
            'prompt_text': 'Classify: "The system shall respond within 2 seconds"',
            'prompt_type': 'zero_shot',
            'length_type': 'short',
            'prompt_complexity': 'low',
            'token_length': 12,
            'inference_time': 10.5,
            'energy': 0.065
        },
        {
            'prompt_text': 'Based on these examples... [detailed context] Now classify this requirement',
            'prompt_type': 'few_shot',
            'length_type': 'medium',
            'prompt_complexity': 'high',
            'token_length': 75,
            'inference_time': 42.1,
            'energy': 0.267
        }
    ]
    
    batch_results = predictor.batch_predict(batch_data)
    
    for i, result in enumerate(batch_results):
        if 'error' not in result:
            print(f"Batch {i+1}: Carbon = {result['predicted_carbon_emission']:.8f}, "
                  f"Efficiency = {result['efficiency_rating']}")
    
    print("\n⚡ SCENARIO 3: Energy Efficiency Analysis")
    print("-" * 40)
    
    # Example 3: Compare efficiency of different prompt strategies
    strategies = [
        ("Zero-shot", "zero_shot", 20, 15.2, 0.089),
        ("Few-shot", "few_shot", 65, 38.7, 0.241), 
        ("Chain-of-Thought", "cot", 40, 28.9, 0.167),
        ("Detailed", "detailed", 85, 47.3, 0.298)
    ]
    
    print("Strategy Comparison:")
    for name, ptype, tokens, time, energy in strategies:
        result = predictor.predict_carbon_emission(
            prompt_text=f"Example {name.lower()} prompt for classification task",
            prompt_type=ptype,
            length_type="medium" if tokens > 30 else "short",
            prompt_complexity="medium",
            token_length=tokens,
            inference_time=time,
            energy=energy
        )
        
        print(f"  {name:15}: Carbon={result['predicted_carbon_emission']:.6f}, "
              f"Efficiency={result['efficiency_rating']:8}, "
              f"Carbon/Token={result['carbon_per_token']:.8f}")
    
    print("\n🎯 SCENARIO 4: Model Information")
    print("-" * 40)
    
    model_info = predictor.get_model_info()
    print(f"Model: {model_info['model_name']}")
    print(f"Total Features: {model_info['num_features']}")
    print(f"Feature Breakdown:")
    for category, count in model_info['feature_categories'].items():
        print(f"  {category}: {count}")
    
    print("\n💡 KEY INSIGHTS:")
    print("="*60)
    print("• Carbon emission = Energy × 0.001 (perfect linear relationship)")
    print("• Energy consumption is the primary driver of carbon footprint")
    print("• Prompt complexity and type affect inference time and energy")
    print("• Shorter, simpler prompts are more carbon-efficient")
    print("• Linear Regression achieved perfect accuracy due to direct relationship")

if __name__ == "__main__":
    main()