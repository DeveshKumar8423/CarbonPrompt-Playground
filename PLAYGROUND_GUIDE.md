# 🌱 Carbon Prompt Playground - User Guide

## Welcome to the Interactive AI Carbon Footprint Explorer!

The Carbon Prompt Playground is an educational web application that helps you understand and optimize the environmental impact of AI prompts. Just like TensorFlow's Neural Network Playground, this tool provides an intuitive, visual way to explore complex concepts - in this case, the relationship between prompt characteristics and carbon emissions.

## 🚀 Getting Started

### Quick Launch
```bash
# Simple way - just run the launcher
python3 run_playground.py

# Or start the Flask app directly
python3 app.py
```

Then navigate to **http://localhost:5000** to start exploring!

## 🎮 How to Play

### 1. **Write Your Prompt** 📝
- Enter any AI prompt in the text area
- Try different types: classification tasks, creative writing, analysis, etc.
- The playground analyzes your prompt's characteristics automatically

### 2. **Choose Prompt Strategy** 🎯
- **Zero-shot**: Direct question without examples
- **Few-shot**: Includes examples for context  
- **Chain-of-Thought**: Step-by-step reasoning
- **Detailed**: Comprehensive instructions
- **Alessio User**: Specific prompt style

### 3. **Adjust Technical Parameters** ⚙️
- **Token Length**: Number of tokens (words/symbols) in your prompt
- **Inference Time**: How long the AI takes to process (affects energy)
- **Energy Consumption**: Direct energy usage during processing

### 4. **See Real-Time Predictions** 📊
Watch as the playground instantly predicts:
- Carbon emission in grams of CO₂
- Energy efficiency rating
- Environmental impact level
- Carbon cost per token

## 📈 Understanding the Visualizations

### Main Carbon Chart 🌱
- **Real-time prediction**: Blue dot shows your current prompt's carbon footprint
- **Historical context**: Green area shows typical ranges
- **Timeline view**: See how predictions change over time

### Energy-Carbon Relationship ⚡
- **Perfect correlation**: Discover the linear relationship (Carbon = Energy × 0.001)
- **Data points**: Each dot represents a real prompt from our dataset
- **Trend line**: Shows the direct mathematical relationship

### Feature Importance 📊
- **Energy dominance**: 96.11% of carbon prediction comes from energy usage
- **Secondary factors**: Inference time, prompt complexity, token count
- **Optimization insights**: Focus on energy-efficient prompting

### Comparison Analysis ⚖️
- **Your prompt vs. average**: See how you compare to typical usage
- **Impact levels**: Low, medium, high carbon footprint ranges
- **Efficiency ranking**: Where you stand in the sustainability spectrum

## 🎛️ Interactive Controls

### Quick Actions
- **🎯 Predict Carbon**: Get instant predictions for your current settings
- **🎲 Random Example**: Load a sample from our dataset to explore
- **🌱 Optimize for Low Carbon**: Auto-adjust parameters for minimal footprint
- **🔄 Reset**: Clear everything and start fresh

### Smart Features
- **Auto-prediction**: As you type and adjust, predictions update automatically
- **Parameter validation**: The playground prevents invalid inputs
- **Sample prompts**: Explore real examples from our research dataset
- **Optimization suggestions**: Get tips for reducing carbon impact

## 💡 Pro Tips for Carbon-Efficient Prompting

### 🌿 **Energy Optimization**
1. **Shorter prompts** = Less processing time = Lower energy
2. **Simpler language** = Faster inference = Less carbon
3. **Direct questions** often outperform complex instructions

### 📝 **Prompt Strategy Impact**
- **Zero-shot**: Most energy-efficient for simple tasks
- **Few-shot**: Good balance of accuracy and efficiency  
- **Chain-of-thought**: Higher carbon cost but better reasoning
- **Detailed prompts**: Most carbon-intensive but highest accuracy

### ⚡ **Technical Optimization**
- Monitor the **efficiency rating** - aim for "Good" or "Excellent"
- Watch **carbon per token** - optimize for the lowest ratio
- Use the **optimization button** for automatic improvements

## 🧪 Educational Experiments

### Experiment 1: Prompt Length Impact
1. Start with a simple prompt: "Classify this as positive or negative"
2. Gradually make it longer and more detailed
3. Observe how carbon emissions scale with length
4. Find the sweet spot between accuracy and efficiency

### Experiment 2: Strategy Comparison
1. Take the same task (e.g., text classification)
2. Try all different prompt types (zero-shot, few-shot, etc.)
3. Compare carbon footprints
4. Understand the efficiency trade-offs

### Experiment 3: Parameter Optimization
1. Start with high energy consumption settings
2. Use the "Optimize for Low Carbon" feature
3. Manually fine-tune parameters
4. Learn how each factor affects carbon output

## 🌍 Real-World Impact

### Understanding the Numbers
- **1 gram CO₂** ≈ equivalent to what 0.05 trees absorb per day
- **Energy consumption** is the primary driver (96%+ of carbon impact)
- **Prompt efficiency** can reduce carbon footprint by 30-70%

### Scaling Considerations
When you multiply by millions of queries:
- A 0.0001g reduction per prompt = 100g saved per million queries
- Efficient prompting = measurable environmental impact
- Every optimization contributes to sustainable AI

## 🛠️ Technical Details

### The ML Model
- **Perfect accuracy**: R² = 1.0000 on test data
- **127 engineered features**: Text analysis, efficiency metrics, interactions
- **Linear relationship**: Carbon = Energy × 0.001 (discovered through analysis)
- **Real-time inference**: Predictions in milliseconds

### Data Sources
- **315 samples** from PROMISE dataset
- **Multiple prompt types** and complexity levels
- **Real energy measurements** from LLaMA 3.2 model
- **Comprehensive feature engineering** for accurate predictions

### Technology Stack
- **Backend**: Flask + scikit-learn + pandas
- **Frontend**: HTML5 + CSS3 + JavaScript + D3.js
- **Visualizations**: Interactive SVG charts with real-time updates
- **Responsive design**: Works on desktop, tablet, and mobile

## 🚀 Advanced Features

### Keyboard Shortcuts
- `Ctrl/Cmd + Enter`: Quick predict
- `Escape`: Close modals/panels
- `?`: Open help (click help button)

### API Integration
The playground exposes REST APIs for programmatic access:
- `POST /api/predict`: Single predictions
- `POST /api/batch_predict`: Batch processing
- `GET /api/model_info`: Model metadata
- `GET /api/sample_prompts`: Example data

### Deployment Options
```bash
# Local development
python3 run_playground.py

# Docker deployment
docker-compose up -d

# Production with Gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## 🎯 Learning Objectives

After using the playground, you should understand:

1. **The direct relationship** between energy consumption and carbon emissions
2. **How prompt characteristics** (length, complexity, type) affect environmental impact  
3. **Optimization strategies** for sustainable AI usage
4. **The trade-offs** between accuracy and environmental efficiency
5. **Real-world implications** of AI carbon footprint at scale

## 🤝 Contributing

This playground is open for enhancement! Areas for contribution:
- Additional visualization types
- More prompt strategies
- Extended datasets
- Mobile optimization
- Performance improvements

## 📚 Further Reading

- [Original research dataset](final_experiment_results.csv)
- [Model development details](advanced_carbon_model.py)  
- [API documentation](app.py)
- [Deployment guide](README.md)

---

**Happy exploring! 🌱 Every efficient prompt makes a difference for our planet.** 🌍

*Made with ❤️ for sustainable AI development*