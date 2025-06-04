import pickle
from scipy.sparse import save_npz, load_npz
import pandas as pd
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity

def find_similar_documents(query, vectorizer, document_vectors, top_n=10):
    # Transform the query using the loaded vectorizer
    query_vector = vectorizer.transform([query])
    
    # Compute cosine similarity between the query vector and all document vectors
    similarities = cosine_similarity(query_vector, document_vectors).flatten()
    
    # Get the top N similar documents
    top_indices = similarities.argsort()[-top_n:][::-1]
    
    return top_indices, similarities[top_indices]

def load_vectorizer_vectors_urls():
    with open('./clustering/clustering_model/tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    document_vectors = load_npz('./clustering/clustering_model/document_vectors.npz')
    urls = pd.read_csv('./clustering/clustering_model/urls.csv')['url'].tolist()
    
    return vectorizer, document_vectors, urls

