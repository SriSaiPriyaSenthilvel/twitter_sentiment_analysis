import streamlit as st
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from PIL import Image

# Sidebar image as navbar
with st.sidebar:
    st.image("logo.png", width=800)  # Replace with your actual image file name
    st.title("Navigation")
    st.markdown("Use this app to predict sentiment from tweets!")

# Load trained model and tokenizer
model = tf.keras.models.load_model('sentiment_rnn_model.h5', compile=False)  # <-- Fix here

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

with open('label_map.pickle', 'rb') as handle:
    label_map = pickle.load(handle)

# Reverse the label map for predictions
inv_label_map = {v: k for k, v in label_map.items()}

MAX_LEN = 60  # Corrected to match training

# App title
st.title("Sentiment Analysis with RNN")
st.subheader("Enter a tweet to predict its sentiment:")

# User input
user_input = st.text_area("Your Tweet:")

if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a tweet.")
    else:
        seq = tokenizer.texts_to_sequences([user_input])
        padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post')
        pred = model.predict(padded)[0]
        label_index = pred.argmax()
        sentiment = inv_label_map[label_index]

        st.success(f"🌟 Predicted Sentiment: **{sentiment}**")
