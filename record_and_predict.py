import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
from speech_emotion import load_speech_model, predict_speech_emotion

def record_audio(filename="live.wav", duration=3, sample_rate=22050):
    print("🎤 Recording... Speak now!")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()
    write(filename, sample_rate, audio)
    print("✔ Recording saved as:", filename)

model = load_speech_model("speech_emotion_model.h5")

def live_speech_emotion():
    record_audio()
    emotion = predict_speech_emotion(model, "live.wav")
    print("🎯 Predicted Emotion:", emotion)

if __name__ == "__main__":
    live_speech_emotion()
