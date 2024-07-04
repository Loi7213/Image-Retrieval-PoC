import os
from config import CONFIG
from feature_extraction import extract_features_and_labels
from knn_clustering import train_knn

def setup_project():
    # Create necessary directories
    os.makedirs(CONFIG['data_path'], exist_ok=True)
    os.makedirs(CONFIG['model_path'], exist_ok=True)

    # Extract features and save
    print("Extracting features...")
    extract_features_and_labels()

    # Train KNN model
    print("Training KNN model...")
    train_knn()

    print("Project setup complete.")

if __name__ == "__main__":
    setup_project()