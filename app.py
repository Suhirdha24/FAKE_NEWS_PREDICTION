from flask import Flask, render_template, request
import pickle
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Initialize Flask app
app = Flask(__name__)

# Load the transformer model and tokenizer for fake news detection
model_name = "distilbert-base-uncased-finetuned-sst-2-english"  # Example model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Save transformer model and tokenizer using pickle (if not done already)
# with open("transformer_model_fake_news.pkl", "wb") as model_file:
#     pickle.dump(model, model_file)
# with open("tokenizer_fake_news.pkl", "wb") as tokenizer_file:
#     pickle.dump(tokenizer, tokenizer_file)

# Example function to use transformer model for fake news detection
def predict_fake_news(news_text):
    # Preprocess and tokenize the input news text
    inputs = tokenizer(
        news_text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    # Get prediction
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

    # Get probabilities for each class (fake or real)
    probs = predictions[0].tolist()

    # Create labels dictionary
    labels = {
        "real_news": probs[0],  # Class 0: real news
        "fake_news": probs[1]   # Class 1: fake news
    }

    # Determine the most likely classification
    max_label = max(labels.items(), key=lambda x: x[1])

    return {
        "prediction": max_label[0],
        "confidence": max_label[1],
        "all_probabilities": labels
    }

# Initialize the app's home route
@app.route('/')
def home():
    return render_template('index.html')

# Modify the /predict route to use the fake news prediction model
@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from the form
    user_input = request.form['content']
    
    # Use the fake news prediction function
    result = predict_fake_news(user_input)
    
    # Extract the prediction and confidence
    prediction = result['prediction']
    confidence = result['confidence']
    
    # Prepare the results to send back to the HTML template
    results = {
        "prediction": prediction,
        "confidence": confidence,
        "all_probabilities": result['all_probabilities']
    }
    
    return render_template('index.html', results=results, user_input=user_input)

if __name__ == '__main__':
    app.run(debug=True)
