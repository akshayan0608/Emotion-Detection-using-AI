import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Flatten

# -------------------------------------------------------
# 1. Feature Extraction (MFCC Sequence)
# -------------------------------------------------------
def extract_mfcc(file_path, max_len=200):
    try:
        audio, sr = librosa.load(file_path, duration=3.0)

        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        mfcc = mfcc.T

        # Pad sequences to same length
        if len(mfcc) < max_len:
            pad_width = max_len - len(mfcc)
            mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')

        else:
            mfcc = mfcc[:max_len]

        return mfcc

    except Exception as e:
        print("Error:", file_path, e)
        return np.zeros((max_len, 40))

# -------------------------------------------------------
# 2. Load Dataset
# -------------------------------------------------------
emotions = ["angry", "happy", "sad", "neutral"]
X, y = [], []

for emotion in emotions:
    folder = f"dataset/{emotion}/"
    for file in os.listdir(folder):
        if file.endswith(".wav"):
            path = folder + file
            features = extract_mfcc(path)
            X.append(features)
            y.append(emotion)

X = np.array(X)
y = LabelEncoder().fit_transform(y)
y = to_categorical(y)

# -------------------------------------------------------
# 3. Train/Test Split
# -------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------------------------------
# 4. CNN + LSTM MODEL (High Accuracy)
# -------------------------------------------------------
model = Sequential()

model.add(Conv1D(128, kernel_size=5, activation='relu', input_shape=(200, 40)))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.3))

model.add(Conv1D(64, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.3))

model.add(LSTM(128, return_sequences=False))
model.add(Dropout(0.3))

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(4, activation='softmax'))

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# -------------------------------------------------------
# 5. Train Model
# -------------------------------------------------------
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=40,
    batch_size=32
)

# -------------------------------------------------------
# 6. Save Model
# -------------------------------------------------------
model.save("speech_emotion_cnn_lstm.h5")
print("✔ High Accuracy Model Saved → speech_emotion_cnn_lstm.h5")
