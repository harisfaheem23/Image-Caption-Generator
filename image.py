import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import pickle
import os

# Define the working directory for loading files
WORKING_DIR = '.'  # Current directory

# Load the trained model
model_path = os.path.join(WORKING_DIR, 'best_model.keras')
tokenizer_path = os.path.join(WORKING_DIR, 'tokenizer.pickle')
max_length_path = os.path.join(WORKING_DIR, 'max_length.pkl')

# Load model, tokenizer, and max_length with error handling
try:
    model = tf.keras.models.load_model(model_path)
except Exception as e:
    st.error(f"Error loading model: {e}")

# Load VGG16 model for feature extraction
vgg_model = VGG16()
vgg_model = tf.keras.Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)

# Load tokenizer and max_length
try:
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
except Exception as e:
    st.error(f"Error loading tokenizer: {e}")

try:
    with open(max_length_path, 'rb') as f:
        max_length = pickle.load(f)
except Exception as e:
    st.error(f"Error loading max length: {e}")

# Helper function to convert index to word
def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# Function to predict image caption
def predict_caption(model, image, tokenizer, max_length):
    in_text = 'startseq'
    for i in range(max_length):
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
    # Remove 'startseq' and 'endseq' from the generated caption
    final_caption = in_text.split()[1:-1]
    return ' '.join(final_caption)

# Main Streamlit app function
def main():
    st.title("Image Caption Generator")
    
    # File uploader for image
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_image is not None:
        # Load and preprocess the uploaded image
        image = load_img(uploaded_image, target_size=(224, 224))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        
        # Feature extraction using VGG16
        feature = vgg_model.predict(image, verbose=0)
        
        # Predict caption using the loaded model
        caption = predict_caption(model, feature, tokenizer, max_length)
        
        # Display the image and predicted caption
        st.image(uploaded_image, caption='Uploaded Image.', use_column_width=True)
        st.write("Generated Caption: ", caption)

if __name__ == "__main__":
    main()
