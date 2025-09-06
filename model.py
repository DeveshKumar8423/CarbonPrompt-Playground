# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

def train_carbon_prediction_model(results_csv_path="results/final_experiment_results.csv"):
    """
    Trains and saves a model to predict carbon emission based on experiment data.
    """
    print("\n" + "="*50)
    print("Training Carbon Emission Prediction Model")
    print("="*50 + "\n")

    # Step 1: Load the data from your experiment results
    try:
        df = pd.read_csv(results_csv_path)
    except FileNotFoundError:
        print(f"Error: The file {results_csv_path} was not found. Please run your experiment first.")
        return

    # Step 2: Prepare the data for the model
    # Features (X) are the things you use to predict
    # The target (y) is the thing you want to predict
    features = ['Token_Length', 'Length_type', 'Prompt_complexity']
    target = 'carbon_emission'
    
    # We need to convert text data (like 'short', 'medium', 'high') into numbers
    # The OneHotEncoder handles this automatically for the 'Length_type' feature
    preprocessor = ColumnTransformer(
        transformers=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'), ['Length_type'])
        ],
        remainder='passthrough'  # Keep the 'Token_Length' and 'Prompt_complexity' columns as they are
    )

    # Step 3: Choose and train the model
    # We'll use Linear Regression as our primary model.
    # The Pipeline combines the preprocessing and the model into one object.
    print("Building a Linear Regression model...")
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])

    # Split the data into training and testing sets (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(
        df[features], df[target], test_size=0.2, random_state=42
    )

    # Train the model on the training data
    pipeline.fit(X_train, y_train)

    # Step 4: Evaluate the model
    # The score tells you how well the model performed on the test data (from 0.0 to 1.0)
    score = pipeline.score(X_test, y_test)
    print(f"Model trained successfully. R-squared score on test data: {score:.4f}")

    # Step 5: Save the trained model to a file
    # joblib is used to save the model so you can use it later in your web server
    model_filename = "carbon_predictor.pkl"
    joblib.dump(pipeline, model_filename)
    print(f"Model saved to {model_filename}")

# Call the function to run the process
train_carbon_prediction_model()