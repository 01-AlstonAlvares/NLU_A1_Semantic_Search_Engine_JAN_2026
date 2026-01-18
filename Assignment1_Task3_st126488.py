import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import time
from flask import Flask, request, render_template_string
from collections import Counter
import nltk
from nltk.corpus import reuters

# DATA INITIALIZATION 
try:
    nltk.download('reuters')
    nltk.download('punkt')
    nltk.download('punkt_tab')
except Exception as e:
    print(f"NLTK Download Error: {e}")

def load_reuters_vocab(max_vocab=2000): # Load Reuters corpus and build vocab
    try:
        raw_sents = reuters.sents(categories='grain')
        clean_corpus = [[w.lower() for w in s if w.isalpha()] for s in raw_sents]
        all_words = [w for s in clean_corpus for w in s]
        counts = Counter(all_words)
        vocab = sorted(counts, key=counts.get, reverse=True)[:max_vocab]
        vocab.append('<UNK>')
        return {w: i for i, w in enumerate(vocab)}, {i: w for i, w in enumerate(vocab)}
    except Exception as e:
        print(f"Error loading Reuters: {e}. Using dummy vocab.")
        d_v = {"love": 0, "sex": 1, "tiger": 2, "cat": 3, "<UNK>": 4}
        d_i = {v: k for k, v in d_v.items()}
        return d_v, d_i

word2index, index2word = load_reuters_vocab()

# MODEL ARCHITECTURE (Skipgram NEG)
class Word2VecNeg(nn.Module):
    def __init__(self, v_size, emb_dim=100):
        super().__init__()
        self.v_embeddings = nn.Embedding(v_size, emb_dim)
        self.u_embeddings = nn.Embedding(v_size, emb_dim)
        self.log_sigmoid = nn.LogSigmoid()
        
    def forward(self, center, target, negative):
        v = self.v_embeddings(center).view(center.size(0), 1, -1)
        u = self.u_embeddings(target).view(target.size(0), 1, -1)
        n = -self.u_embeddings(negative)
        pos = torch.bmm(u, v.transpose(1, 2)).view(center.size(0), -1)
        neg = torch.bmm(n, v.transpose(1, 2)).squeeze(2)
        return -torch.mean(self.log_sigmoid(pos) + torch.sum(self.log_sigmoid(neg), 1))

# Instantiate model
model = Word2VecNeg(len(word2index), 100)

# RETRIEVAL & SIMILARITY LOGIC 

def retrieve_top_10_similar(query_word):
    """Computes dot product to retrieve top 10 most similar contexts."""
    if query_word not in word2index:
        return None
    
    # Get vector and all model vectors
    query_idx = word2index[query_word]
    all_vectors = model.v_embeddings.weight.detach().numpy()
    query_vec = all_vectors[query_idx]
    
    # Compute dot products with the corpus 
    dot_products = np.dot(all_vectors, query_vec)
    
    # Normalize for cosine similarity scores
    norms = np.linalg.norm(all_vectors, axis=1) * np.linalg.norm(query_vec)
    similarities = dot_products / (norms + 1e-10)
    
    # Sort and get top 10 (excluding the word itself)
    sorted_indices = np.argsort(similarities)[::-1]
    top_indices = [idx for idx in sorted_indices if idx != query_idx][:10]
    
    return [{"word": index2word[idx], "score": round(float(similarities[idx] * 10), 2)} for idx in top_indices]

def get_similarity(w1, w2): # Computes similarity score between two words.
    if w1 == w2: return 10.0 # Identical words
    if w1 not in word2index or w2 not in word2index: return "N/A"
    
    v1 = model.v_embeddings.weight[word2index[w1]].detach().numpy()
    v2 = model.v_embeddings.weight[word2index[w2]].detach().numpy()
    
    cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
    return round((cos_sim + 1) * 5, 2)

# FLASK INTERFACE
app = Flask(__name__)

CSV_FILE = 'combined.csv'
human_lookup = {}
if os.path.exists(CSV_FILE): # Load human similarity scores
    try:
        df_human = pd.read_csv(CSV_FILE)
        human_lookup = {(str(row['Word 1']).lower(), str(row['Word 2']).lower()): row['Human (mean)'] 
                        for _, row in df_human.iterrows()}
    except Exception as e: print(f"Error reading CSV: {e}")

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Psycholinguistic Experiment</title>
    <style>
        body { font-family: sans-serif; max-width: 700px; margin: 50px auto; padding: 20px; background: #f4f7f6; }
        .card { background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h2 { color: #2c3e50; margin-top: 0; }
        input { display: block; width: 95%; padding: 10px; margin: 10px 0; border: 1px solid #ddd; border-radius: 5px; }
        button { background: #3498db; color: white; border: none; padding: 12px; width: 100%; border-radius: 5px; cursor: pointer; font-size: 16px; margin-top: 10px; }
        .res { margin-top: 20px; padding: 15px; background: #e8f4fd; border-left: 5px solid #3498db; }
        .similar-list { list-style: none; padding: 0; }
        .similar-list li { background: #fff; margin: 5px 0; padding: 8px; border-radius: 4px; border: 1px solid #eee; display: flex; justify-content: space-between; }
    </style>
</head>
<body>
    <div class="card">
        <h2>Word Similarity Estimator</h2>
        <p>Instructions: Assign a score from 0 to 10.</p>
        <form method="POST">
            <input type="text" name="name" placeholder="Your Full Name" required>
            <input type="text" name="w1" placeholder="Word 1" required>
            <input type="text" name="w2" placeholder="Word 2" required>
            <button type="submit">Submit Estimation</button>
        </form>

        {% if word1 %}
        <div class="res">
            <p><strong>Participant:</strong> {{ name }}</p>
            <p><strong>Similarity ({{ word1 }} & {{ word2 }}):</strong></p>
            <ul>
                <li>AI Score: {{ ai_score }}</li>
                <li>Human Score: {{ human_score }}</li>
            </ul>
            
            <p><strong>Top 10 Most Similar Contexts (for {{ word1 }}):</strong></p>
            <ul class="similar-list">
                {% if top_similar %}
                    {% for item in top_similar %}
                    <li><span>{{ item.word }}</span> <span>{{ item.score }}</span></li>
                    {% endfor %}
                {% else %}
                    <li>Word not in corpus.</li>
                {% endif %}
            </ul>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def home():
    data = {}
    if request.method == 'POST':
        name = request.form.get('name') # Get participant name
        w1 = request.form.get('w1').strip().lower()
        w2 = request.form.get('w2').strip().lower()
        
        ai_val = get_similarity(w1, w2)
        h_val = human_lookup.get((w1, w2), human_lookup.get((w2, w1), "Data Not Found"))
        top_10 = retrieve_top_10_similar(w1)
        
        data = {'name': name, 'word1': w1, 'word2': w2, 'ai_score': ai_val, 'human_score': h_val, 'top_similar': top_10} 
        
    return render_template_string(HTML, **data)

if __name__ == '__main__':
    app.run(debug=True, port=5003)