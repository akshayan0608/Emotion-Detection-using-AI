from speech_emotion import load_speech_model, predict_speech_emotion

model = load_speech_model("speech_emotion_model.h5")

audio_file = "test.wav"   # put your audio file name here

emotion = predict_speech_emotion(model, audio_file)
print("Predicted Speech Emotion:", emotion)
