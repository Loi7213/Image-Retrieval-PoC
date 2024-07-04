import pickle
from sklearn.neighbors import NearestNeighbors
from config import CONFIG
import os
import joblib

def train_knn():
    # Load embeddings
    with open(os.path.join(CONFIG['model_path'], CONFIG['embeddings_file']), 'rb') as f:
        features = pickle.load(f)
    
    # Ensure features is 2D array
    if len(features.shape) == 1:
        features = features.reshape(-1, 1)
        
    knn = NearestNeighbors(n_neighbors=CONFIG['num_neighbors'], metric='cosine')
    knn.fit(features)
    
    joblib.dump(knn, os.path.join(CONFIG['model_path'], CONFIG['knn_model_file']))
    print(f"KNN model trained and saved in {CONFIG['model_path']}.")

if __name__ == "__main__":
    os.makedirs(CONFIG['model_path'], exist_ok=True)
    train_knn()