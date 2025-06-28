from flask import Flask, render_template, request
import joblib
import re
import string

app = Flask(__name__)

# Load model
model = joblib.load('model.pkl')

# Cleaning function (same as training)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        news = request.form['news']
        clean_news = clean_text(news)
        result = model.predict([clean_news])[0]
        prediction = "Fake News" if result == "FAKE" else "Real News"
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
