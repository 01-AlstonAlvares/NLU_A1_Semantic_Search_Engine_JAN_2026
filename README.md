# 🧠 Psycholinguistic Word Embedding Analysis & Semantic Search Engine

---

## 📌 Project Overview

This project explores the **development**, **training**, and **psycholinguistic evaluation** of three foundational **word embedding architectures**:

- **Skip-gram (Softmax)**
- **Skip-gram (Negative Sampling)**
- **GloVe (Global Vectors)**

The work is organized into **three structured tasks**, progressing from **core model implementation**, to **quantitative benchmarking against human judgment**, and finally to an **interactive psycholinguistic experiment interface**.

> ✨ *The goal is to understand how well computational word representations align with human semantic intuition.*

---

## 📂 Task 1: Word Embedding Implementation

In this task, all models were implemented **from scratch in PyTorch**, strictly following the methodologies outlined in the reference notebooks.

### 🔹 Skip-gram (Softmax)

- Predicts **context words** given a **center word**.
- Uses a **single embedding matrix**.
- Employs a **linear output layer** followed by **`CrossEntropyLoss`**.
- Suitable for smaller vocabularies but computationally expensive for large corpora.

### 🔹 Skip-gram with Negative Sampling (NEG)

- An optimized variant of Skip-gram.
- Uses **two embedding matrices**:
  - `v` → center word embeddings
  - `u` → context word embeddings
- Converts a **multiclass prediction** problem into **binary classification**.
- Trained using **`LogSigmoid` loss** to:
  - Maximize similarity for true word-context pairs
  - Minimize similarity for randomly sampled *negative* words

### 🔹 GloVe (Global Vectors)

- A **log-bilinear regression model**.
- Trained on a **global word–word co-occurrence matrix**.
- Learns embeddings by minimizing a **weighted least-squares objective**.
- Captures both **local context** and **global corpus statistics**.

---

## 📂 Task 2: Performance & Correlation Analysis

This task evaluates how well the learned embeddings reflect **semantic and syntactic relationships**, and how closely they align with **human similarity judgments**.

### 🔸 Dynamic Windowing

- A flexible training function allows dynamic adjustment of `window_size` (e.g., **2, 5, 10**).
- Larger windows:
  - Capture broader semantic context
  - Increase training time
- Smaller windows:
  - Focus on syntactic relationships

### 🔸 Analogy Evaluation

Models were evaluated using **vector arithmetic** on:

- **Syntactic analogies**
  - Example: *walk : walked :: run : ?*
- **Semantic analogies**
  - Example: *Athens : Greece :: Paris : ?*

### 🔸 Human Judgment Correlation

- Used the **WordSim353** dataset (`combined.csv`).
- Computed **Spearman Rank Correlation** between:
  - Model-based similarity (dot product)
  - Human-assigned similarity scores

### 🔸 Benchmarking

- Compared custom-trained embeddings against a **pre-trained baseline**:
  - **GloVe (Gensim)** → `glove-wiki-gigaword-100`

---

## 📂 Task 3: Psycholinguistic Experiment Interface

A **Flask-based web application** was developed to simulate a **digital psycholinguistic word similarity experiment**.

### 🌐 Key Features

- **Digital Similarity Estimation** (0.0 – 10.0 scale)
- **Psycholinguistic Consistency** (antonyms treated as similar)
- **Top-10 Semantic Retrieval** using dot product similarity

---

## 📊 Data Sources & Tech Stack

### 📚 Data Sources

- **Reuters-21578 Corpus** (grain category)
- **WordSim353 (`combined.csv`)**
- **Google Analogy Dataset**

### 🛠️ Tech Stack

| Category | Tools |
|--------|------|
| Neural Networks | PyTorch |
| Web Framework | Flask |
| NLP Utilities | NLTK, Gensim |
| Statistics | SciPy |
| Data Handling | Pandas, NumPy |

---

## 🚀 How to Run

```bash
pip install torch flask pandas numpy nltk gensim scipy requests
```

Run benchmarks, then start Flask and visit:
http://127.0.0.1:5003

---

✨ **End of README** ✨
