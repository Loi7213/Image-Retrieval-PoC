import pickle
from sklearn.neighbors import NearestNeighbors
from config import CONFIG
import os
import joblib

def train_knn():
    # Load embeddings
    try:
        with open(os.path.join(CONFIG['model_path'], CONFIG['embeddings_file']), 'rb') as f:
            features = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{CONFIG['embeddings_file']}' was not found in '{CONFIG['model_path']}'.")
        return
    except (pickle.UnpicklingError, EOFError):
        print(f"Error: The file '{CONFIG['embeddings_file']}' is corrupted or in an invalid format.")
        return

    # Ensure features is 2D array
    if len(features.shape) == 1:
        features = features.reshape(-1, 1)
    elif features.shape[0] < 1:
        print("Error: The features array is empty. There must be at least one sample to train the KNN model.")
        return

    knn = NearestNeighbors(n_neighbors=CONFIG['num_neighbors'], metric='cosine')
    knn.fit(features)

    try:
        os.makedirs(CONFIG['model_path'], exist_ok=True)
        joblib.dump(knn, os.path.join(CONFIG['model_path'], CONFIG['knn_model_file']))
        print(f"KNN model trained and saved in {CONFIG['model_path']}.")
    except (OSError, IOError) as e:
        print(f"Error: Failed to save the KNN model. {e}")

if __name__ == "__main__":
    os.makedirs(CONFIG['model_path'], exist_ok=True)
    train_knn()