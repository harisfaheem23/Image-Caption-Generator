import os
import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import pickle

# Ensure TensorFlow runs only on the CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Define working directory
WORKING_DIR = '.'

# Load model, tokenizer, and max_length with proper error handling
model_path = os.path.join(WORKING_DIR, 'best_model.keras')
tokenizer_path = os.path.join(WORKING_DIR, 'tokenizer.pickle')
max_length_path = os.path.join(WORKING_DIR, 'max_length.pkl')

try:
    model = tf.keras.models.load_model(model_path)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

try:
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
except Exception as e:
    st.error(f"Error loading tokenizer: {e}")
    st.stop()

try:
    with open(max_length_path, 'rb') as f:
        max_length = pickle.load(f)
except Exception as e:
    st.error(f"Error loading max_length: {e}")
    st.stop()

# Load VGG16 model for feature extraction
try:
    vgg_model = VGG16(weights='imagenet', include_top=False)
    vgg_model = tf.keras.Model(inputs=vgg_model.input, outputs=tf.keras.layers.GlobalAveragePooling2D()(vgg_model.output))
except Exception as e:
    st.error(f"Error loading VGG16 model: {e}")
    st.stop()

# Helper function to convert index to word
def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# Function to generate a caption for an image
def predict_caption(model, image, tokenizer, max_length):
    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = tf.keras.preprocessing.sequence.pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([image, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat, tokenizer)
        if word is None:
            break
        in_text += " " + word
        if word == 'endseq':
            break
    return ' '.join(in_text.split()[1:-1])

# Streamlit app main function
def main():
    st.title("Image Caption Generator")
    st.write("Upload an image to generate a caption.")

    # File uploader
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        # Load and preprocess the image
        try:
            image = load_img(uploaded_image, target_size=(224, 224))
            image = img_to_array(image)
            image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
            image = preprocess_input(image)
            
            # Extract features using VGG16
            features = vgg_model.predict(image, verbose=0)
            
            # Generate caption
            caption = predict_caption(model, features, tokenizer, max_length)
            
            # Display the image and caption
            st.image(uploaded_image, caption='Uploaded Image.', use_column_width=True)
            st.write("Generated Caption:", caption)
        except Exception as e:
            st.error(f"Error processing the image: {e}")

if __name__ == "__main__":
    main()
