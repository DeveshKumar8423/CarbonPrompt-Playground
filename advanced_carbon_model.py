#!/usr/bin/env python3
"""
Advanced Carbon Emission Prediction Model
==========================================

This module creates a comprehensive ML pipeline to predict carbon emissions 
from prompt characteristics using multiple algorithms and advanced feature engineering.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class CarbonEmissionPredictor:
    """
    Advanced Carbon Emission Prediction Model with comprehensive feature engineering
    and ensemble methods.
    """
    
    def __init__(self, data_path="final_experiment_results.csv"):
        """Initialize the predictor with data path."""
        self.data_path = data_path
        self.models = {}
        self.best_model = None
        self.feature_importance = None
        self.scaler = StandardScaler()
        
    def load_and_preprocess_data(self):
        """Load and preprocess the data with advanced feature engineering."""
        print("\n" + "="*60)
        print("LOADING AND PREPROCESSING DATA")
        print("="*60)
        
        # Load data
        self.df = pd.read_csv(self.data_path)
        print(f"Loaded {len(self.df)} samples with {len(self.df.columns)} features")
        
        # Advanced Feature Engineering
        print("Engineering advanced features...")
        
        # 1. Text features from prompts
        self.df['prompt_word_count'] = self.df['Prompt_used'].str.split().str.len()
        self.df['prompt_char_count'] = self.df['Prompt_used'].str.len()
        self.df['prompt_avg_word_length'] = self.df['prompt_char_count'] / self.df['prompt_word_count']
        self.df['prompt_has_quotes'] = self.df['Prompt_used'].str.contains('"').astype(int)
        self.df['prompt_has_question'] = self.df['Prompt_used'].str.contains('\\?').astype(int)
        self.df['prompt_complexity_score'] = (
            self.df['prompt_word_count'] * 0.3 + 
            self.df['prompt_char_count'] * 0.001 + 
            self.df['Token_Length'] * 0.7
        )
        
        # 2. Efficiency ratios
        self.df['energy_per_token'] = self.df['energy'] / (self.df['Token_Length'] + 1)
        self.df['carbon_per_token'] = self.df['carbon_emission'] / (self.df['Token_Length'] + 1)
        self.df['energy_efficiency'] = self.df['Token_Length'] / (self.df['inference_time'] + 0.001)
        
        # 3. Interaction features
        complexity_map = {'low': 1, 'medium': 2, 'high': 3}
        self.df['complexity_numeric'] = self.df['Prompt_complexity'].map(complexity_map)
        length_map = {'short': 1, 'medium': 2}
        self.df['length_numeric'] = self.df['Length_type'].map(length_map)
        
        self.df['complexity_length_interaction'] = (
            self.df['complexity_numeric'] * self.df['length_numeric']
        )
        
        # 4. Response quality features
        self.df['response_length'] = self.df['raw_response'].str.len()
        self.df['response_word_count'] = self.df['raw_response'].str.split().str.len()
        
        print(f"Created {len(self.df.columns) - 14} new engineered features")
        
        # Prepare features for modeling
        self.prepare_features()
        
    def prepare_features(self):
        """Prepare feature matrices including text vectorization."""
        print("Preparing feature matrices...")
        
        # Categorical features to encode
        categorical_features = ['Prompt_type', 'Length_type', 'Prompt_complexity']
        
        # Numerical features
        numerical_features = [
            'Token_Length', 'inference_time', 'energy', 'iteration_number',
            'prompt_word_count', 'prompt_char_count', 'prompt_avg_word_length',
            'prompt_has_quotes', 'prompt_has_question', 'prompt_complexity_score',
            'energy_per_token', 'energy_efficiency', 'complexity_numeric',
            'length_numeric', 'complexity_length_interaction', 'response_length',
            'response_word_count'
        ]
        
        # Text vectorization for prompts
        print("Vectorizing prompt text...")
        tfidf = TfidfVectorizer(
            max_features=100,  # Limit features due to small dataset
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        
        prompt_tfidf = tfidf.fit_transform(self.df['Prompt_used'])
        tfidf_feature_names = [f'tfidf_{i}' for i in range(prompt_tfidf.shape[1])]
        
        # Create TF-IDF DataFrame
        tfidf_df = pd.DataFrame(
            prompt_tfidf.toarray(),
            columns=tfidf_feature_names,
            index=self.df.index
        )
        
        # Combine all features
        self.X = pd.concat([
            self.df[numerical_features],
            pd.get_dummies(self.df[categorical_features], prefix=categorical_features),
            tfidf_df
        ], axis=1)
        
        self.y = self.df['carbon_emission']
        
        print(f"Final feature matrix: {self.X.shape}")
        print(f"   - Numerical features: {len(numerical_features)}")
        print(f"   - Categorical features: {sum(self.X.columns.str.startswith(tuple(categorical_features)))}")
        print(f"   - Text features: {len(tfidf_feature_names)}")
        
    def train_multiple_models(self):
        """Train multiple models and compare their performance."""
        print("\n" + "="*60)
        print("TRAINING MULTIPLE ML MODELS")
        print("="*60)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.df['Prompt_type']
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Define models to test
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=0.001),
            'ElasticNet': ElasticNet(alpha=0.001, l1_ratio=0.5),
            'Random Forest': RandomForestRegressor(
                n_estimators=100, 
                max_depth=10, 
                random_state=42,
                min_samples_split=5
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=100, 
                max_depth=6, 
                random_state=42,
                learning_rate=0.1
            ),
            'SVR': SVR(kernel='rbf', C=1.0, gamma='scale'),
            'Neural Network': MLPRegressor(
                hidden_layer_sizes=(100, 50),
                max_iter=500,
                random_state=42,
                early_stopping=True
            )
        }
        
        # Train and evaluate each model
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Use scaled data for models that benefit from it
            if name in ['SVR', 'Neural Network', 'Lasso Regression', 'ElasticNet']:
                X_train_model = X_train_scaled
                X_test_model = X_test_scaled
            else:
                X_train_model = X_train
                X_test_model = X_test
            
            # Train model
            model.fit(X_train_model, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_model)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Cross-validation score
            if name in ['SVR', 'Neural Network', 'Lasso Regression', 'ElasticNet']:
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
            else:
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            
            results[name] = {
                'model': model,
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'cv_r2_mean': cv_scores.mean(),
                'cv_r2_std': cv_scores.std(),
                'y_pred': y_pred
            }
            
            print(f"   R² Score: {r2:.4f}")
            print(f"   RMSE: {rmse:.6f}")
            print(f"   CV R² Mean: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        self.results = results
        self.X_test = X_test
        self.y_test = y_test
        
        # Find best model
        best_model_name = max(results, key=lambda x: results[x]['cv_r2_mean'])
        self.best_model_name = best_model_name
        self.best_model = results[best_model_name]['model']
        
        print(f"\nBEST MODEL: {best_model_name}")
        print(f"   R² Score: {results[best_model_name]['r2']:.4f}")
        print(f"   CV R² Mean: {results[best_model_name]['cv_r2_mean']:.4f}")
        
        return results
        
    def create_ensemble_model(self):
        """Create an ensemble model from the best performing individual models."""
        print("\n" + "="*60)
        print("CREATING ENSEMBLE MODEL")
        print("="*60)
        
        # Select top 3 models based on CV score
        sorted_models = sorted(
            self.results.items(), 
            key=lambda x: x[1]['cv_r2_mean'], 
            reverse=True
        )[:3]
        
        print("🔝 Top 3 models for ensemble:")
        ensemble_models = []
        for i, (name, result) in enumerate(sorted_models, 1):
            print(f"   {i}. {name}: CV R² = {result['cv_r2_mean']:.4f}")
            ensemble_models.append((name.lower().replace(' ', '_'), result['model']))
        
        # Create voting ensemble
        ensemble = VotingRegressor(estimators=ensemble_models)
        
        # Train ensemble
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.df['Prompt_type']
        )
        
        ensemble.fit(X_train, y_train)
        y_pred_ensemble = ensemble.predict(X_test)
        
        # Evaluate ensemble
        ensemble_r2 = r2_score(y_test, y_pred_ensemble)
        ensemble_rmse = np.sqrt(mean_squared_error(y_test, y_pred_ensemble))
        
        print(f"\n🎯 ENSEMBLE PERFORMANCE:")
        print(f"   R² Score: {ensemble_r2:.4f}")
        print(f"   RMSE: {ensemble_rmse:.6f}")
        
        # Update best model if ensemble is better
        if ensemble_r2 > self.results[self.best_model_name]['r2']:
            print(f"   Ensemble outperforms best individual model!")
            self.best_model = ensemble
            self.best_model_name = "Ensemble"
        
        return ensemble
    
    def analyze_feature_importance(self):
        """Analyze feature importance for tree-based models."""
        print("\n" + "="*60)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*60)
        
        # Get feature importance from Random Forest
        if 'Random Forest' in self.results:
            rf_model = self.results['Random Forest']['model']
            feature_importance = pd.DataFrame({
                'feature': self.X.columns,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("Top 15 Most Important Features (Random Forest):")
            for i, row in feature_importance.head(15).iterrows():
                print(f"   {row['feature']}: {row['importance']:.4f}")
            
            self.feature_importance = feature_importance
            
            # Plot feature importance
            plt.figure(figsize=(12, 8))
            top_features = feature_importance.head(20)
            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Feature Importance')
            plt.title('Top 20 Feature Importance (Random Forest)')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
            print("   Feature importance plot saved as 'feature_importance.png'")
    
    def hyperparameter_tuning(self):
        """Perform hyperparameter tuning for the best models."""
        print("\n" + "="*60)
        print(" HYPERPARAMETER TUNING")
        print("="*60)
        
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.df['Prompt_type']
        )
        
        # Tune Random Forest
        print("🌲 Tuning Random Forest...")
        rf_params = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf_grid = GridSearchCV(
            RandomForestRegressor(random_state=42),
            rf_params,
            cv=5,
            scoring='r2',
            n_jobs=-1,
            verbose=0
        )
        rf_grid.fit(X_train, y_train)
        
        print(f"   Best RF params: {rf_grid.best_params_}")
        print(f"   Best RF CV score: {rf_grid.best_score_:.4f}")
        
        # Tune Gradient Boosting
        print("Tuning Gradient Boosting...")
        gb_params = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'min_samples_split': [2, 5, 10]
        }
        
        gb_grid = GridSearchCV(
            GradientBoostingRegressor(random_state=42),
            gb_params,
            cv=5,
            scoring='r2',
            n_jobs=-1,
            verbose=0
        )
        gb_grid.fit(X_train, y_train)
        
        print(f"   Best GB params: {gb_grid.best_params_}")
        print(f"   Best GB CV score: {gb_grid.best_score_:.4f}")
        
        # Update best model if tuned version is better
        tuned_models = {
            'Tuned Random Forest': rf_grid.best_estimator_,
            'Tuned Gradient Boosting': gb_grid.best_estimator_
        }
        
        best_tuned_score = max(rf_grid.best_score_, gb_grid.best_score_)
        if best_tuned_score > self.results[self.best_model_name]['cv_r2_mean']:
            if rf_grid.best_score_ > gb_grid.best_score_:
                self.best_model = rf_grid.best_estimator_
                self.best_model_name = "Tuned Random Forest"
            else:
                self.best_model = gb_grid.best_estimator_
                self.best_model_name = "Tuned Gradient Boosting"
            
            print(f"   Tuned model is now the best: {self.best_model_name}")
        
        return tuned_models
    
    def generate_predictions_report(self):
        """Generate a comprehensive predictions report."""
        print("\n" + "="*60)
        print("GENERATING PREDICTIONS REPORT")
        print("="*60)
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame({
            'Model': list(self.results.keys()),
            'R² Score': [self.results[model]['r2'] for model in self.results.keys()],
            'RMSE': [self.results[model]['rmse'] for model in self.results.keys()],
            'MAE': [self.results[model]['mae'] for model in self.results.keys()],
            'CV R² Mean': [self.results[model]['cv_r2_mean'] for model in self.results.keys()],
            'CV R² Std': [self.results[model]['cv_r2_std'] for model in self.results.keys()]
        }).sort_values('CV R² Mean', ascending=False)
        
        print("MODEL PERFORMANCE COMPARISON:")
        print(comparison_df.to_string(index=False, float_format='%.4f'))
        
        # Save detailed results
        comparison_df.to_csv('model_comparison_results.csv', index=False)
        print("\nDetailed results saved to 'model_comparison_results.csv'")
        
        return comparison_df
    
    def save_best_model(self, filename="best_carbon_predictor.pkl"):
        """Save the best model and preprocessing pipeline."""
        print(f"\nSaving best model ({self.best_model_name}) to {filename}...")
        
        # Create a complete pipeline
        pipeline_data = {
            'model': self.best_model,
            'scaler': self.scaler,
            'feature_columns': list(self.X.columns),
            'model_name': self.best_model_name,
            'preprocessing_steps': {
                'categorical_features': ['Prompt_type', 'Length_type', 'Prompt_complexity'],
                'numerical_features': [col for col in self.X.columns if not col.startswith(('Prompt_type', 'Length_type', 'Prompt_complexity', 'tfidf_'))],
                'tfidf_features': [col for col in self.X.columns if col.startswith('tfidf_')]
            }
        }
        
        joblib.dump(pipeline_data, filename)
        print(f"Model saved successfully!")
        
        return filename
    
    def run_complete_pipeline(self):
        """Run the complete ML pipeline."""
        print("STARTING ADVANCED CARBON EMISSION ML PIPELINE")
        print("="*80)
        
        # Step 1: Load and preprocess data
        self.load_and_preprocess_data()
        
        # Step 2: Train multiple models
        self.train_multiple_models()
        
        # Step 3: Analyze feature importance
        self.analyze_feature_importance()
        
        # Step 4: Create ensemble model
        self.create_ensemble_model()
        
        # Step 5: Hyperparameter tuning
        self.hyperparameter_tuning()
        
        # Step 6: Generate report
        comparison_df = self.generate_predictions_report()
        
        # Step 7: Save best model
        self.save_best_model()
        
        print("\n" + "="*80)
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"🏆 Best Model: {self.best_model_name}")
        print("="*80)
        
        return self.best_model, comparison_df

def main():
    """Main function to run the carbon emission prediction pipeline."""
    predictor = CarbonEmissionPredictor("final_experiment_results.csv")
    best_model, results = predictor.run_complete_pipeline()
    return predictor, best_model, results

if __name__ == "__main__":
    predictor, best_model, results = main()