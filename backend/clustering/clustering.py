import json
import time
import pickle
from scipy.sparse import save_npz, load_npz
import fastcluster
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import dendrogram

def read_json_file(filepath):
    with open(filepath, 'r', encoding="ISO-8859-1") as file:
        return json.load(file)

def filter_and_prepare_data(data):
    url_list = []
    document_list = []

    # Check if 'response' and 'docs' are present in the data
    if 'response' in data and 'docs' in data['response']:
        for response_val in data['response']['docs']:
            # Validate that both 'url' and 'content' are present and meet the conditions
            if 'url' in response_val and 'content' in response_val:
                if response_val['url'] not in url_list and len(response_val['content'].split()) > 3500:
                    url_list.append(response_val['url'])
                    document_list.append(response_val['content'])
    return url_list, document_list

def vectorize_texts(document_list, url_list):
    vectorizer = TfidfVectorizer(max_df=0.75, min_df=0.1, stop_words='english', use_idf=True, max_features=100000, dtype=np.float32)
    X = vectorizer.fit_transform(document_list)
    with open('./data/tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    save_vectorizer_vectors_urls(vectorizer, X, url_list)
    return X

def perform_kmeans(X, num_clusters):
    km = KMeans(n_clusters=num_clusters, init='k-means++', max_iter=100, n_init=1)
    km.fit(X)
    return km.labels_

def save_results(ids, labels, filename):
    results_df = pd.DataFrame({'id': ids, 'cluster': labels})
    results_df.to_csv(filename, sep=',', header=False, index=False, encoding='utf-8')

def hierarchical_clustering(X, method='ward'):
    dist = 1 - cosine_similarity(X)
    linkage_matrix = fastcluster.linkage(dist, method=method, metric='euclidean')
    return linkage_matrix

def plot_dendrogram(linkage_matrix, labels, method='ward'):
    
    fig, ax = plt.subplots()
    try:
        ax = dendrogram(linkage_matrix, orientation="right", labels=labels)
    except ValueError as e:
        print(e)
        print(f"Expected number of labels: {len(linkage_matrix) + 1}")

    # Get labels
    for key in ax:
        if key == "ivl":
            hc_key = ax[key]
        if key == "color_list":
            hc_dict = dict([(y,x+1) for x,y in enumerate(sorted(set(ax[key])))])
            hc_value = [hc_dict[x] for x in ax[key]]

    # Store hierarchical clustering results in a file
    hc_cluster_series = pd.Series(hc_value)
    hc_id_series = pd.Series(hc_key)
    hc_results = (pd.concat([hc_id_series, hc_cluster_series], axis=1))
    hc_results.columns = ['id', 'cluster']
    hc_results.to_csv(f"clustering_h_{method}.txt", sep=',', columns=['id', 'cluster'], header=False, index=False, encoding='utf-8')

def setup_directory(directory='./data'):
    if not os.path.exists(directory):
        os.makedirs(directory)
        
def save_vectorizer_vectors_urls(vectorizer, document_vectors, urls):
    setup_directory('./data')
    
    # Save the vectorizer
    with open('./data/tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    
    # Save the document vectors
    save_npz('./data/document_vectors.npz', document_vectors)
    
    # Save the URLs in a CSV for easy access and alignment
    url_df = pd.DataFrame(urls, columns=['url'])
    url_df.to_csv('./data/urls.csv', index=False, header=True)

def load_vectorizer_vectors_urls():
    with open('./data/tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    document_vectors = load_npz('./data/document_vectors.npz')
    urls = pd.read_csv('./data/urls.csv')['url'].tolist()
    
    return vectorizer, document_vectors, urls

def find_similar_documents(query, vectorizer, document_vectors, top_n=10):
    # Transform the query using the loaded vectorizer
    query_vector = vectorizer.transform([query])
    
    # Compute cosine similarity between the query vector and all document vectors
    similarities = cosine_similarity(query_vector, document_vectors).flatten()
    
    # Get the top N similar documents
    top_indices = similarities.argsort()[-top_n:][::-1]
    
    return top_indices, similarities[top_indices]



def main():
    start_time = time.time()
    setup_directory()
    data = read_json_file('solr_data.json')
    url_list, document_list = filter_and_prepare_data(data)
    X = vectorize_texts(document_list, url_list)
    labels = perform_kmeans(X, 11)
    save_results(url_list, labels, './data/clustering_f.txt')
    ward_linkage_matrix = hierarchical_clustering(X, method='ward')
    plot_dendrogram(ward_linkage_matrix, url_list, method='ward')
    single_linkage_matrix = hierarchical_clustering(X, method='single')
    plot_dendrogram(single_linkage_matrix, url_list, method='single')

    print("Total time taken: ", time.time() - start_time)
    
    # Test
    # vectorizer, document_vectors, urls = load_vectorizer_vectors_urls()
    # top_indices, top_similarities = find_similar_documents('aluminum', vectorizer=vectorizer, document_vectors=document_vectors, top_n=10)
    # results = [{'url': urls[index], 'similarity': top_similarities[i]} for i, index in enumerate(top_indices)]
    # print(results)
    
if __name__ == "__main__":
    main()
