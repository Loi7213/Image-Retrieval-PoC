import streamlit as st
from PIL import Image
import os
from image_search import load_models, search_similar_images
from config import CONFIG

st.set_page_config(page_title="Car Image Search", layout="wide")

@st.cache_resource
def load_search_models():
    return load_models()

resnet_model, knn_model = load_search_models()

st.title("Car Image Search")

uploaded_file = st.file_uploader("Choose a car image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    if st.button('Search Similar Images'):
        similar_images, similar_labels = search_similar_images(image, resnet_model, knn_model)
        
        st.subheader("Similar Images:")
        cols = st.columns(3)
        for idx, (img_path, label) in enumerate(zip(similar_images, similar_labels)):
            with cols[idx % 3]:
                st.image(img_path, caption=f"Label: {label}", use_column_width=True)

st.sidebar.text("Car Image Search App")
st.sidebar.info("Upload an image of a car to find similar images in our database.")

if __name__ == "__main__":
    import socket
    socket.setdefaulttimeout(30)  # Set default timeout to 30 seconds