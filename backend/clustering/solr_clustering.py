
import json
import time
import pickle
import fastcluster
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import ward, dendrogram

start_time = time.time()

# Open SOLR Index JSON file (Get your SOLR response JSON file here, file too large to upload to GitHub)
f = open('solr_data.json', encoding="ISO-8859-1")
data = json.load(f)
f.close()

document_list = []
url_list = []

# Parse text content from indexed json
for outer_index in data:
    if outer_index == "response":
        response_val = data[outer_index]
        for curr_key in response_val:
            if curr_key == "docs":
                site_info = response_val[curr_key]
                for site_dict in site_info:
                    if site_dict.get("url") \
                            and site_dict.get("content") \
                            and site_dict['url'] not in url_list \
                            and len(site_dict['content'].split()) > 3500:
                        url_list.append(site_dict['url'])
                        document_list.append(site_dict['content'])
print("Time taken for parsing JSON: ", time.time() - start_time)
#print(document_list[0])

# Use TF-IDF Vectorizer to vectorize document text inputs
vectorizer = TfidfVectorizer(max_df=0.75, min_df=0.1, stop_words='english', use_idf=True, max_features=100000, dtype=np.float32)
X = vectorizer.fit_transform(document_list)
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
del vectorizer
print("Time taken for vectorizing inputs: ", time.time() - start_time)

# Apply flat clustering (K-means)
km = KMeans(n_clusters=11, init='k-means++', max_iter=100, n_init=1)
km.fit(X)
print("Time taken for applying flat clustering: ", time.time() - start_time)

# Store K-means clustering results in a file
id_series = pd.Series(url_list)
cluster_series = pd.Series(km.labels_)
results = (pd.concat([id_series,cluster_series], axis=1))
results.columns = ['id', 'cluster']
results.to_csv("clustering_f.txt", sep=',', columns=['id', 'cluster'], header=False, index=False, encoding='utf-8')
print("Time taken for storing results of flat clustering: ", time.time() - start_time)

# Apply Hierarchical Clustering (Single link)
dist = 1 - cosine_similarity(X)
del X
print("Time taken for computing cosine similarity: ", time.time() - start_time)

# Single Linkage
agg_d = fastcluster.linkage(dist, method='single', metric='euclidean')
print("Time taken for single linkage: ", time.time() - start_time)

fig, ax = plt.subplots()
try:
    ax = dendrogram(agg_d, orientation="right", labels=url_list)
except ValueError as e:
    print(e)
    print(f"Expected number of labels: {len(agg_d) + 1}")
print("Time taken for applying hierarchical clustering: ", time.time() - start_time)

# Get labels
for key in ax:
    if key == "ivl":
        hc_key = ax[key]
    if key == "color_list":
        hc_dict = dict([(y,x+1) for x,y in enumerate(sorted(set(ax[key])))])
        hc_value = [hc_dict[x] for x in ax[key]]
print("Time taken for getting labels: ", time.time() - start_time)

# Store hierarchical clustering results in a file
hc_cluster_series = pd.Series(hc_value)
hc_id_series = pd.Series(hc_key)
hc_results = (pd.concat([hc_id_series, hc_cluster_series], axis=1))
hc_results.columns = ['id', 'cluster']
hc_results.to_csv("clustering_h_single.txt", sep=',', columns=['id', 'cluster'], header=False, index=False, encoding='utf-8')

print("Time taken for storing results of single linkage hierarchical clustering: ", time.time() - start_time)

# Alternative linkage
agg_d = fastcluster.linkage(dist, method='ward', metric='euclidean')
print("Time taken for single linkage: ", time.time() - start_time)

fig, ax = plt.subplots()
try:
    ax = dendrogram(agg_d, orientation="right", labels=url_list)
except ValueError as e:
    print(e)
    print(f"Expected number of labels: {len(agg_d) + 1}")
print("Time taken for applying hierarchical clustering: ", time.time() - start_time)

# Get labels
for key in ax:
    if key == "ivl":
        hc_key = ax[key]
    if key == "color_list":
        hc_dict = dict([(y,x+1) for x,y in enumerate(sorted(set(ax[key])))])
        hc_value = [hc_dict[x] for x in ax[key]]
print("Time taken for getting labels: ", time.time() - start_time)

# Store hierarchical clustering results in a file
hc_cluster_series = pd.Series(hc_value)
hc_id_series = pd.Series(hc_key)
hc_results = (pd.concat([hc_id_series, hc_cluster_series], axis=1))
hc_results.columns = ['id', 'cluster']
hc_results.to_csv("clustering_h_ward.txt", sep=',', columns=['id', 'cluster'], header=False, index=False, encoding='utf-8')

print("Time taken for storing results of ward linkage hierarchical clustering: ", time.time() - start_time)