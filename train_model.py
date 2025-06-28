import pandas as pd 
import numpy as np
import re
import string
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("fake_or_real_news.csv")

# Preprocessing
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['text'] = df['text'].apply(clean_text)

# Splitting dataset
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# Creating pipeline (TF-IDF + Logistic Regression)
model = make_pipeline(TfidfVectorizer(), LogisticRegression())

# Training model
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'C:/Users/VAISHNAVI/Desktop/Project/model.pkl')

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
