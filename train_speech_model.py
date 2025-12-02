# train_speech_model.py
import os
import numpy as np
import librosa
import soundfile
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

# -------------------------------------------------------
# 1. SAFE FEATURE EXTRACTION (NO MEMORY ERROR)
# -------------------------------------------------------
def extract_features(file_name):
    try:
        # Load only first 3 seconds - prevents RAM overload
        audio_data, sample_rate = librosa.load(file_name, sr=None, duration=3.0)

        # Pad to 3 seconds if short
        required_len = int(sample_rate * 3)
        if len(audio_data) < required_len:
            audio_data = np.pad(audio_data, (0, required_len - len(audio_data)), mode='constant')

        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)
        mfcc_mean = np.mean(mfcc.T, axis=0)
        return mfcc_mean

    except Exception as e:
        print("Feature extraction error:", e)
        return np.zeros(40)


# -------------------------------------------------------
# 2. LOAD DATASET (SORTED FOLDERS)
# -------------------------------------------------------
emotions = ['angry', 'happy', 'sad', 'neutral']
features = []
labels = []

for emotion in emotions:
    folder = f"dataset/{emotion}/"
    if not os.path.exists(folder):
        print(f"WARNING: Folder missing → {folder}")
        continue

    for file in os.listdir(folder):
        if file.endswith(".wav"):
            file_path = folder + file
            data = extract_features(file_path)
            features.append(data)
            labels.append(emotion)

X = np.array(features)
y = LabelEncoder().fit_transform(labels)
y = to_categorical(y)

print("Dataset Loaded!")
print("Samples:", len(X))


# -------------------------------------------------------
# 3. TRAIN / TEST SPLIT
# -------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# -------------------------------------------------------
# 4. MODEL ARCHITECTURE
# -------------------------------------------------------
model = Sequential([
    Dense(256, activation='relu', input_shape=(40,)),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(4, activation='softmax')
])

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)


# -------------------------------------------------------
# 5. TRAIN MODEL
# -------------------------------------------------------
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32
)


# -------------------------------------------------------
# 6. SAVE MODEL
# -------------------------------------------------------
model.save("speech_emotion_model.h5")
print("✔ MODEL SAVED → speech_emotion_model.h5")
