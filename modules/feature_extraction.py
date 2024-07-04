import numpy as np
import os
import pickle
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
from config import CONFIG

def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def extract_features_and_labels():
    model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    
    features = []
    labels = []
    image_paths = []
    
    for folder in ['car_train', 'car_test']:
        folder_path = os.path.join(CONFIG['data_path'], folder)
        if os.path.exists(folder_path):
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(root, file)
                        img = load_and_preprocess_image(img_path)
                        feature = model.predict(img)
                        features.append(feature.flatten())
                        image_paths.append(img_path)
                        # Assuming label is the parent directory name
                        label = os.path.basename(root)
                        labels.append(label)
        else:
            print(f"Warning: Folder {folder_path} does not exist.")
    
    return np.array(features), labels, image_paths

if __name__ == "__main__":
    features, labels, image_paths = extract_features_and_labels()
    os.makedirs(CONFIG['model_path'], exist_ok=True)
    
    # Save embeddings
    with open(os.path.join(CONFIG['model_path'], CONFIG['embeddings_file']), 'wb') as f:
        pickle.dump(features, f)
    
    # Save labels
    with open(os.path.join(CONFIG['model_path'], CONFIG['labels_file']), 'wb') as f:
        pickle.dump(labels, f)
    
    # Save image paths
    np.save(os.path.join(CONFIG['model_path'], CONFIG['image_paths_file']), image_paths)
    
    print(f"Features, labels, and image paths saved in {CONFIG['model_path']}. Total images processed: {len(image_paths)}")