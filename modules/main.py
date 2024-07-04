import os
from config import CONFIG
from feature_extraction import extract_features_and_labels
from knn_clustering import train_knn

def setup_project():
    print("Starting setup...")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Data path: {CONFIG['data_path']}")
    print(f"Model path: {CONFIG['model_path']}")

    # Create necessary directories
    os.makedirs(CONFIG['data_path'], exist_ok=True)
    print(f"Created data directory: {CONFIG['data_path']}")
    os.makedirs(CONFIG['model_path'], exist_ok=True)
    print(f"Created model directory: {CONFIG['model_path']}")

    # Extract features and save
    print("Extracting features...")
    extract_features_and_labels()

    # Train KNN model
    print("Training KNN model...")
    train_knn()

    print("Project setup complete.")

if __name__ == "__main__":
    setup_project()