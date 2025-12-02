import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import os

# Sample dataset
data = {
    'text': [
        "I am very happy today",
        "I feel so sad",
        "I am angry at you",
        "I am surprised",
        "I am scared",
        "I feel neutral"
    ],
    'label': ['happy', 'sad', 'angry', 'surprise', 'fear', 'neutral']
}

df = pd.DataFrame(data)

# Create models folder if not exist
if not os.path.exists("models"):
    os.makedirs("models")

# Build and train model
pipe = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression(max_iter=1000))
])

pipe.fit(df['text'], df['label'])

# Save trained model
joblib.dump(pipe, 'models/text_model.joblib')
print("Model saved as models/text_model.joblib")
