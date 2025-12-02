# speech_emotion.py
import numpy as np
import librosa
import soundfile
from tensorflow.keras.models import load_model

# -------------------------------------------------------
# 1. SAFE FEATURE EXTRACTION (NO MEMORY ERROR)
# -------------------------------------------------------
def extract_features(file_name):
    try:
        # Load ONLY first 3 seconds - prevents huge RAM usage
        audio_data, sample_rate = librosa.load(file_name, sr=None, duration=3.0)

        # Pad short audio to fixed 3 seconds
        required_len = int(sample_rate * 3)
        if len(audio_data) < required_len:
            padding = required_len - len(audio_data)
            audio_data = np.pad(audio_data, (0, padding), mode='constant')

        # Extract MFCC (40 features)
        mfcc = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)
        mfcc_mean = np.mean(mfcc.T, axis=0)

        return mfcc_mean

    except Exception as e:
        print(f"Error extracting features from {file_name}: {e}")
        return np.zeros(40)


# -------------------------------------------------------
# 2. LOAD SPEECH EMOTION MODEL
# -------------------------------------------------------
def load_speech_model(model_path="speech_emotion_model.h5"):
    try:
        model = load_model(model_path)
        print("Speech Emotion Model loaded successfully.")
        return model
    except Exception as e:
        print("Error loading speech model:", e)
        return None


# -------------------------------------------------------
# 3. PREDICT EMOTION FROM AUDIO FILE
# -------------------------------------------------------
emotion_labels = ['angry', 'happy', 'sad', 'neutral']

def predict_speech_emotion(model, audio_file):
    if model is None:
        return "Model not loaded"

    try:
        features = extract_features(audio_file)
        features = features.reshape(1, -1)

        prediction = model.predict(features)
        emotion = emotion_labels[np.argmax(prediction)]

        return emotion

    except Exception as e:
        print("Prediction error:", e)
        return "Error"
