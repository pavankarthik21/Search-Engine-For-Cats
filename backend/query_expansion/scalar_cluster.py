"""
Author: Sharon T Alexander
"""

from nltk.stem import PorterStemmer
from collections import defaultdict
import itertools
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from solr_client import search
import pysolr
import re

from nltk.corpus import stopwords

import nltk
nltk.download('words')
from nltk.corpus import words
word_list = set(words.words())

def is_real_word(word):
    return word in word_list


def remove_stopwords_nltk(query, language='english'):
    stop_words = set(stopwords.words(language))
    words = query.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)


def tokenize_doc(doc_text, stop_words):
    text = doc_text
    text = text.strip("\\")
    text = re.sub(r'[\n]', ' ', text)
    text = re.sub(r'[,-]', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub('[0-9]', '', text)
    text = text.lower()
    tkns = text.split(' ')
    tokens = [token for token in tkns if token not in stop_words and token != '' and not token.isnumeric()]
    return tokens


def complete_word(stem, words):
    for word in words:
        if word.startswith(stem):
            return word
    return 'None'


def build_metric_correlation(documents):
    """Create metric correlation matrix using term proximity"""
    res = []
    collected_result = {}
    for result in documents:
        if 'title' in result:
            if result['title'] not in collected_result:
                collected_result[result['title']] = 1
                res.append(result)

    print("QE V3")

    # re-assign
    documents = res

    stemmer = PorterStemmer()
    temp_correlation = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    correlation = defaultdict(lambda: defaultdict(float))
    stop_words = set(stopwords.words('english'))
    all_tokens = set()

    for response in documents:
        if "title" in response and "content" in response:
            doc = response["title"] + " " + response["content"]
            tokens = tokenize_doc(doc, stop_words)
            stems = []
            for t in tokens:
                all_tokens.add(t)
                stems.append(stemmer.stem(t))
            positions = defaultdict(list)

            # Record positions for each stem
            for pos, stem in enumerate(stems):
                positions[stem].append(pos)

            # Calculate inverse distance for all stem pairs
            for a, b in itertools.combinations(positions.keys(), 2):
                total = 0
                for i in positions[a]:
                    for j in positions[b]:
                        total += 1 / (abs(i - j) + 1)  # +1 prevents division by zero
                avg_total = total / (len(positions[a]) * len(positions[b]))
                temp_correlation[a][b]['avg_total'] += avg_total
                temp_correlation[a][b]['doc_count'] += 1
                temp_correlation[b][a]['avg_total'] += avg_total
                temp_correlation[b][a]['doc_count'] += 1

            for a, b in itertools.combinations(positions.keys(), 2):
                correlation[a][b] = temp_correlation[a][b]['avg_total']/temp_correlation[a][b]['doc_count']
                correlation[b][a] = temp_correlation[a][b]['avg_total'] / temp_correlation[a][b]['doc_count']


    return correlation, all_tokens


def extract_words(word_list, all_tokens, exclude_words=[], k=2):
    neighbors = []
    for tup in word_list:
        if tup[0].strip() and complete_word(tup[0], all_tokens) != 'None':
            # only consider words and not empty spaces
            if len(neighbors) == k:
                break
            if tup[0] not in exclude_words:
                neighbors.append(tup[0])
    return neighbors


def get_expanded_query(query, query_stems, expansion_scores, all_tokens, k=2):
    ex_qry = []
    for word in query_stems:
        word_list = sorted(expansion_scores[word].items(), key=lambda x: x[1], reverse=True)
        ex_qry.extend(extract_words(word_list, all_tokens, exclude_words=ex_qry[:] + query_stems[:], k=k))
        # complete the stems
        expanded_query = ''

    for word in ex_qry:
        if not is_real_word(word):
            expanded_query += ' ' + complete_word(word, all_tokens)
        else:
            expanded_query += ' ' + word
    expanded_query = query + expanded_query
    return expanded_query


def expand_query_with_scalar_clustering(query, documents, n_terms=3):
    """
    Full pipeline for scalar clustering-based query expansion

    Args:
        documents: List of text documents
        query: Original search query
        n_terms: Number of expansion terms to add

    Returns:
        Dictionary with original query, expansion terms, and expanded query
    """
    # 1. Build metric correlation matrix
    metric_corr, all_tokens = build_metric_correlation(documents)

    # 2. Prepare for scalar clustering
    stems = list(metric_corr.keys())
    if not stems:
        return {"expanded_query": query}  # No terms found

    # Create correlation vectors
    vectors = []
    for stem in stems:
        vectors.append([metric_corr[stem].get(other, 0) for other in stems])
    vectors = np.array(vectors)

    # 3. Compute cosine similarities
    sim_matrix = cosine_similarity(vectors)

    # 4. Process query terms
    stemmer = PorterStemmer()
    query_stems = list(set([stemmer.stem(t) for t in query.lower().split()]))

    # 5. Aggregate similarity scores for expansion candidates
    expansion_scores = defaultdict(lambda : defaultdict(float))
    for q_stem in query_stems:
        try:
            idx = stems.index(q_stem)
        except ValueError:
            continue  # Skip unknown query terms

        # Accumulate scores for all other stems
        for other_idx, score in enumerate(sim_matrix[idx]):
            if stems[other_idx] != q_stem:  # Exclude self
                expansion_scores[q_stem][stems[other_idx]] += score

    # 6. Select top terms not already in query
    expanded_query = get_expanded_query(query, query_stems, expansion_scores, all_tokens, k=2)

    return expanded_query


if __name__ == '__main__':
    # Connect to your Solr core (replace with your actual Solr URL and core name)
    solr = pysolr.Solr('http://localhost:8983/solr/nutch', timeout=10, always_commit=True)

    while True:
        query = input('Enter a query: ')

        if query == 'q!':
            print("Exiting QE...")
            break

        # search results
        original_query = query
        results = search(server=solr, query=original_query)

        print("Initial Results:")
        # Print each result's fields
        for idx, result in enumerate(results):
            print(f'{idx}: {result["title"]}')
        if results:
            expanded_query = expand_query_with_scalar_clustering(remove_stopwords_nltk(query), results)
            print("Expanded query terms:", expanded_query)

            qe_results = search(server=solr, query=expanded_query)

            print("QE Results:")
            # Print each result's fields
            for result in qe_results:
                print(result["title"])
        else:
            print("No results found.")
