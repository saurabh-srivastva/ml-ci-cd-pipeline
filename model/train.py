# train.py

import joblib
import os
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier  # Changed this import

def train_and_save_model():
    """
    Trains a simple model on the Iris dataset and saves it to disk.
    """
    # Load the Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Train a RandomForestClassifier model
    model = RandomForestClassifier(n_estimators=100, random_state=42) # Changed the model
    model.fit(X, y)
    print("Model trained successfully.")

    # --- Saving the model ---
    # Define the directory and model path
    model_dir = 'model'
    model_path = os.path.join(model_dir, 'iris_model.pkl')

    # Create the directory if it doesn't exist
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"Directory '{model_dir}' created.")

    # Save the trained model to the specified path
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == '__main__':
    train_and_save_model()
