import numpy as np
import pickle
import os
import joblib
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
from config import CONFIG

def load_models():
    resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    knn_model = joblib.load(os.path.join(CONFIG['model_path'], CONFIG['knn_model_file']))
    return resnet_model, knn_model

def search_similar_images(img, resnet_model, knn_model):
    # Preprocess the input image
    img = img.resize((224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    # Extract features
    feature = resnet_model.predict(x)
    
    # Find similar images
    _, indices = knn_model.kneighbors(feature.reshape(1, -1))
    
    # Load image paths and labels
    image_paths = np.load(os.path.join(CONFIG['model_path'], CONFIG['image_paths_file']))
    with open(os.path.join(CONFIG['model_path'], CONFIG['labels_file']), 'rb') as f:
        labels = pickle.load(f)
    
    similar_images = [image_paths[i] for i in indices[0]]
    similar_labels = [labels[i] for i in indices[0]]
    
    return similar_images, similar_labels