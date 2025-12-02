from text_emotion import load_text_emotion_model, get_text_emotion

model = load_text_emotion_model("models/text_model.joblib")

text = input("Enter text: ")

emotion, scores = get_text_emotion(model, text)

print("Predicted Emotion:", emotion)
print("Scores:", scores)
