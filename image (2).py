import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
import os

# Load models and tokenizer
# Adjust the WORKING_DIR for your deployment environment
WORKING_DIR = r'C:\Users\VICTUS\latest'

try:
    model = tf.keras.models.load_model(os.path.join(WORKING_DIR, 'best_model1.keras'))
except Exception as e:
    st.error(f"Error loading model: {e}")

# Load VGG16 model for feature extraction
vgg_model = VGG16(weights='imagenet', include_top=False, pooling='avg')
vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.output)

# Load tokenizer and max_length
try:
    with open(os.path.join(WORKING_DIR, 'tokenizer.pickle'), 'rb') as f:
        tokenizer = pickle.load(f)
    with open(os.path.join(WORKING_DIR, 'max_length.pkl'), 'rb') as f:
        max_length = pickle.load(f)
except Exception as e:
    st.error(f"Error loading tokenizer or max_length: {e}")

# Function to convert index to word
def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# Function to predict caption
def predict_caption(model, image, tokenizer, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([image, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat, tokenizer)
        if word is None:
            break
        if word == 'endseq':  # Stop if 'endseq' is encountered
            break
        in_text += " " + word  # Append word to caption

    # Remove 'startseq' from the start of the caption
    caption = in_text.replace('startseq ', '').strip()

    return caption

# Streamlit app main function
def main():
    st.title("Image Caption Generator")
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_image is not None:
        # Load and preprocess the image
        image = load_img(uploaded_image, target_size=(224, 224))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        
        # Extract features
        try:
            feature = vgg_model.predict(image, verbose=0)
            # Generate caption
            caption = predict_caption(model, feature, tokenizer, max_length)
            
            # Display image and caption
            st.image(uploaded_image, caption='Uploaded Image.', use_column_width=True)
            st.write(f"**Generated Caption:** {caption}")
        except Exception as e:
            st.error(f"Error processing image: {e}")

if __name__ == "__main__":
    main()
