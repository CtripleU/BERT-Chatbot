from flask import Flask, request, jsonify, render_template
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import os

app = Flask(__name__)

# Load the model and tokenizer
model_save_path = r'C:\Users\Lenovo\OneDrive\Downloads\BERT-Chatbot\saved-model'
tokenizer = BertTokenizer.from_pretrained(model_save_path)
model = BertForSequenceClassification.from_pretrained(model_save_path, local_files_only=True)

# Ensure the model is in evaluation mode
model.eval()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    instruction = request.form['instruction']
    input_text = request.form['input']

    combined_text = instruction + " " + input_text

    # Tokenize the input text
    inputs = tokenizer(combined_text, return_tensors='pt', padding='max_length', truncation=True, max_length=512)
    input_ids = inputs['input_ids']

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=-1).item()

    response = {'prediction': prediction}
    return render_template('index.html', response=response['prediction'])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
