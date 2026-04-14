import streamlit as st
import tensorflow as tf
import librosa
import numpy as np
import os

# --- 1. MODEL ARCHITECTURE (The "Skeleton") ---
def build_model_skeleton(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same'),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same'),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Conv2D(128, (2, 2), activation='relu'),
        tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding='same'),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model

# --- 2. SAFE LOADING ---
@st.cache_resource
def load_my_model():
    # Note: Using the shape from your data.json (130 time steps, 13 MFCCs)
    input_shape = (130, 13, 1) 
    model = build_model_skeleton(input_shape)
    model.load_weights("music_genre_model.h5")
    return model

model = load_my_model()
GENRES = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]

# --- 3. THE PREDICTION FUNCTION (This fixes your NameError) ---
def predict_genre(file_path):
    # Load audio
    signal, sr = librosa.load(file_path, sr=22050)
    
    # Extract MFCCs for the first 3 seconds to match training
    # 3 seconds * 22050 Hz = 66150 samples
    mfcc = librosa.feature.mfcc(y=signal[:66150], sr=sr, n_mfcc=13, n_fft=2048, hop_length=512)
    mfcc = mfcc.T
    
    # Ensure it's the right length (130)
    if len(mfcc) > 130:
        mfcc = mfcc[:130]
    elif len(mfcc) < 130:
        mfcc = np.pad(mfcc, ((0, 130 - len(mfcc)), (0, 0)), mode='constant')

    # Reshape for CNN: (1, 130, 13, 1)
    mfcc = mfcc[np.newaxis, ..., np.newaxis]
    
    # Predict
    prediction = model.predict(mfcc)
    predicted_index = np.argmax(prediction, axis=1)
    return GENRES[predicted_index[0]], prediction[0]

# --- 4. STREAMLIT UI ---
st.title("🎵 Music Genre Classifier")
st.write("Upload a .wav file to identify its genre!")

uploaded_file = st.file_uploader("Choose a .wav file...", type=["wav"])

if uploaded_file is not None:
    # Save the file temporarily
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.audio("temp.wav")
    
    if st.button("Classify Genre"):
        with st.spinner('Analyzing...'):
            # Catch BOTH returned values here
            result, probabilities = predict_genre("temp.wav") 
            
            st.success(f"The predicted genre is: **{result.upper()}**")
            
            # Now create the chart using 'probabilities' (instead of 'mfcc')
            import pandas as pd
            chart_data = pd.DataFrame({
                'Genre': GENRES,
                'Confidence': probabilities
            })
            st.bar_chart(chart_data.set_index('Genre'))
            
            st.balloons()

            # Optional cleanup

if os.path.exists("temp.wav"):
    os.remove("temp.wav")