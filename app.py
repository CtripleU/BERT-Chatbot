from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, BertForSequenceClassification
import torch
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
import os

app = Flask(__name__)

# Load your saved model and tokenizer
model_path = "./saved_model"
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load the sentence transformer model
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

# Try to load label_mapping
try:
    with open('label_mapping.json', 'r') as f:
        label_mapping = json.load(f)
except FileNotFoundError:
    print("label_mapping.json not found. Please ensure it was saved during training.")
    label_mapping = {}

# Reverse the label_mapping for easy lookup
rev_label_mapping = {v: k for k, v in label_mapping.items()}

# Encode all possible responses
response_embeddings = semantic_model.encode(list(label_mapping.keys()))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def semantic_search(query, top_k=5):
    query_embedding = semantic_model.encode([query])
    similarities = cosine_similarity(query_embedding, response_embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    return [(list(label_mapping.keys())[i], similarities[i]) for i in top_indices]

def predict_category(instruction, input_text):
    cleaned_instruction = clean_text(instruction)
    cleaned_input = clean_text(input_text)

    inputs = tokenizer(f"{cleaned_instruction} {cleaned_input}", return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = torch.argmax(logits, dim=-1).item()
    bert_prediction = rev_label_mapping[predicted_class_id]

    semantic_results = semantic_search(f"{cleaned_instruction} {cleaned_input}")

    if bert_prediction in [result[0] for result in semantic_results]:
        return bert_prediction
    else:
        return semantic_results[0][0]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    instruction = request.form['instruction']
    user_input = request.form['user_input']
    response = predict_category(instruction, user_input)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)