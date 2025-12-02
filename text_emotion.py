import joblib
import numpy as np

# Load the saved model
def load_text_emotion_model(model_path):
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        print("Error loading model:", e)
        return None

# Predict emotion
def get_text_emotion(model, text):
    if model is None:
        return "Model not loaded", {}

    # Convert to feature (simple example — depends on your model)
    try:
        prediction = model.predict([text])[0]
        probabilities = model.predict_proba([text])[0]

        # Create dictionary for scores
        emotion_scores = {}
        for idx, emotion in enumerate(model.classes_):
            emotion_scores[emotion] = float(probabilities[idx])

        return prediction, emotion_scores

    except Exception as e:
        print("Error during prediction:", e)
        return "Error", {}
