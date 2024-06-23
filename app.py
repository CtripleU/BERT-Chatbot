from flask import Flask, request, jsonify, render_template
import torch
from transformers import AutoTokenizer, BertForSequenceClassification
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import re
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load the label mapping
with open('./label_mapping.json', 'r') as file:
    label_mapping = json.load(file)

# Load the BERT model and tokenizer
model = BertForSequenceClassification.from_pretrained('./saved_model')
tokenizer = AutoTokenizer.from_pretrained('./saved_model')

# Load the semantic model and response embeddings
semantic_model = SentenceTransformer('./semantic_model')
response_embeddings = np.load('./response_embeddings.npy')

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
    bert_prediction = list(label_mapping.keys())[list(label_mapping.values()).index(predicted_class_id)]
    semantic_results = semantic_search(f"{cleaned_instruction} {cleaned_input}")
    if bert_prediction in [result[0] for result in semantic_results]:
        return bert_prediction
    else:
        return semantic_results[0][0]

@app.route('/')
def home():
    return render_template('index.html')

# @app.route('/')
# def index():
#     return send_from_directory(app.static_folder, 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    instruction = data.get('instruction')
    input_text = data.get('input')
    if not instruction or not input_text:
        return jsonify({'error': 'Invalid input'}), 400
    response = predict_category(instruction, input_text)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
