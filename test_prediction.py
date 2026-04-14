import tensorflow as tf
import numpy as np
import librosa

# Load the brain
model = tf.keras.models.load_model("music_genre_model.h5")
genres = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]

def predict(file_path):
    # Load and process the audio
    signal, sr = librosa.load(file_path, sr=22050)
    # Extract MFCCs for a 3-second segment
    mfcc = librosa.feature.mfcc(y=signal[:66150], sr=sr, n_mfcc=13, n_fft=2048, hop_length=512)
    mfcc = mfcc.T[np.newaxis, ..., np.newaxis] # Reshape for CNN
    
    prediction = model.predict(mfcc)
    result = genres[np.argmax(prediction)]
    return result

# Use a file from your dataset to test
print(f"Predicted Genre: {predict('data/genres_original/rock/rock.00001.wav')}")