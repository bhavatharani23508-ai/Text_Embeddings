# Sentence Embedding and Semantic Similarity

## Overview

This project demonstrates how to convert sentences into numerical vector representations called **embeddings** and measure the semantic similarity between sentences using **cosine similarity**.

The project uses the `all-MiniLM-L6-v2` model from Sentence Transformers to generate sentence embeddings.

## Features

* Converts text sentences into embeddings
* Displays the embedding dimension
* Calculates cosine similarity between sentences
* Identifies semantically similar sentences
* Uses a pretrained Sentence Transformer model

## Technologies Used

* Python
* Sentence Transformers
* Scikit-learn
* all-MiniLM-L6-v2

## Project Structure

```text
Embedding-model/
│
├── app.py
├── venv/
└── README.md
```

## Installation

Create and activate a virtual environment:

```bash
python -m venv venv
```

Activate the virtual environment on Windows:

```bash
venv\Scripts\activate
```

Install the required libraries:

```bash
pip install sentence-transformers scikit-learn
```

## How to Run

Run the following command in the VS Code terminal:

```bash
python app.py
```

The program generates embeddings for the given sentences and calculates their semantic similarity.

## How It Works

### 1. Sentence Embedding

The `all-MiniLM-L6-v2` model converts each sentence into a numerical vector.

```python
embeddings = model.encode(sentences)
```

Each sentence is represented by a vector with 384 dimensions.

### 2. Cosine Similarity

Cosine similarity is used to compare the embeddings:

```python
similarity = cosine_similarity(embeddings)
```

A higher similarity value indicates that two sentences are more semantically related.

### 3. Similarity Filtering

The project displays sentence pairs whose similarity score is greater than `0.5`.

```python
if similarity[i][j] > 0.5:
```

## Example

For example:

```text
Sentence 1: I love eating pizza.
Sentence 2: Pizza is my favorite food.
Similarity: 0.82
```

These sentences have a high similarity because they discuss a similar topic.

## Applications

Sentence embeddings and semantic similarity can be used in:

* Text search
* Question answering systems
* Recommendation systems
* Document comparison
* Chatbots
* Duplicate text detection
* Natural Language Processing applications
