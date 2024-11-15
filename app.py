from flask import Flask, render_template, request
import pickle
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import CountVectorizer

# Initialize Flask app
app = Flask(__name__)

# Load the model and tokenizer from pickle files
with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("count_vectorizer.pkl", "rb") as vec_file:
    vectorizer = pickle.load(vec_file)

# Load the model from Hugging Face if needed
tokenizer = AutoTokenizer.from_pretrained("cybersectony/phishing-email-detection-distilbert_v2.4.1")
model_transformer = AutoModelForSequenceClassification.from_pretrained("cybersectony/phishing-email-detection-distilbert_v2.4.1")

# Define the prediction route
@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        # Get the input text from the user
        news_text = request.form["news_text"]
        
        # Vectorize the text using CountVectorizer (if applicable)
        news_vectorized = vectorizer.transform([news_text])

        # Make predictions using the model
        with torch.no_grad():
            inputs = tokenizer(news_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
            outputs = model_transformer(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(predictions, dim=1).item()
            
            # Define your class labels
            class_labels = {0: 'Fake', 1: 'Real'}
            
            result = class_labels[predicted_class]
            confidence = predictions[0][predicted_class].item()
            
            return render_template("index.html", prediction=result, confidence=confidence)

    return render_template("index.html", prediction=None)

if __name__ == "__main__":
    app.run(debug=True)
