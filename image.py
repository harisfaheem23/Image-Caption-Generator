import streamlit as st
import os
import pickle as p
import numpy as np
import re
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences

base_path = "."

# Load pre-trained VGG16 model for feature extraction
def load_vgg16():
    vgg_model = VGG16()
    return Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)

# Load model, tokenizer, and other resources
def load_resources():
    file_paths = {
        'features': os.path.join(base_path, 'features.pkl'),
        'max_length': os.path.join(base_path, 'max_length.pkl'),
        'tokenizer': os.path.join(base_path, 'tokenizer.pickle'),
        'model': os.path.join(base_path, 'best_model.keras')
    }
    
    # Check if all files exist
    for key, path in file_paths.items():
        if not os.path.exists(path):
            st.error(f"{key} file not found at {path}")
            raise FileNotFoundError(f"{key} file not found at {path}")
    
    # Load files
    with open(file_paths['features'], 'rb') as f:
        features = p.load(f)
    with open(file_paths['max_length'], 'rb') as f:
        max_length = p.load(f)
    with open(file_paths['tokenizer'], 'rb') as f:
        tokenizer = p.load(f)
    model = tf.keras.models.load_model(file_paths['model'])
    return features, max_length, tokenizer, model

# Clean the captions
def clean_caption(caption):
    caption = caption.lower()
    caption = re.sub(r'[^\w\s]', '', caption)
    caption = re.sub(r'\s+', ' ', caption)
    return 'startseq ' + ' '.join([word for word in caption.split() if len(word) > 1]) + ' endseq'

# Predict caption for a given image
def predict_caption(model, image, tokenizer, max_length):
    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], max_length)
        yhat = model.predict([image, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat, tokenizer)
        if word is None:
            break
        in_text += " " + word
        if word == 'endseq':
            break
    return in_text.replace('startseq', '').replace('endseq', '').strip()

# Helper function to convert index to word
def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# Streamlit UI
st.title('Image Captioning')
st.write('Upload an image to get a caption.')

uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    # Load and preprocess image
    image = Image.open(uploaded_file)
    image = image.resize((224, 224))
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    
    # Load resources
    try:
        features, max_length, tokenizer, model = load_resources()
    except FileNotFoundError as e:
        st.error(f"Error loading resources: {e}")
        st.stop()
    
    # Extract features and predict caption
    vgg_model = load_vgg16()
    feature = vgg_model.predict(image, verbose=0)
    caption = predict_caption(model, feature, tokenizer, max_length)
    
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
    st.write(f'Predicted Caption: {caption}')'''

with open("features.pkl", "rb") as f:
    features = pickle.load(f)
print(features)
