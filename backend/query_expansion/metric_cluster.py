"""
Author: Sharon T Alexander
"""

from nltk.stem import PorterStemmer
from collections import defaultdict
import itertools
from solr_client import search
import pysolr
from nltk.corpus import stopwords
import re

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

def metric_cluster_main(query, results, n_neighbors=3):
    res = []
    collected_result = {}
    for result in results:
        if 'title' in result:
            if result['title'] not in collected_result:
                collected_result[result['title']] = 1
                res.append(result)

    print("QE V3")

    # re-assign
    results = res

    stemmer = PorterStemmer()
    correlation = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    stop_words = set(stopwords.words('english'))
    all_tokens = set()

    # 1. Process documents to build correlation matrix
    for response in results:
        print(response)
        if 'title' in response and 'content' in response:
            doc = response["title"] + " " + response["content"]
            tokens = tokenize_doc(doc, stop_words)
            stems = []
            for t in tokens:
                all_tokens.add(t)
                stems.append(stemmer.stem(t))


        positions = defaultdict(list)

        for pos, stem in enumerate(stems):
            positions[stem].append(pos)

        for a, b in itertools.combinations(positions.keys(), 2):
            total = 0
            for i in positions[a]:
                for j in positions[b]:
                    total += 1 / (abs(i - j) + 1)  # +1 to avoid division by zero
            # take the average of total
            avg_total = total/(len(positions[a])*len(positions[b]))
            correlation[a][b]['avg_total'] += avg_total
            correlation[a][b]['doc_count'] += 1
            correlation[b][a]['avg_total'] += avg_total
            correlation[b][a]['doc_count'] += 1

    # 2. Process query and find expansion terms
    query_terms = query.lower().split()
    query_stems = list(set([stemmer.stem(t) for t in query_terms]))  # deduplicate

    expansion_terms = defaultdict(float)

    for stem in query_stems:
        if stem not in correlation:
            continue

        # Get top neighbors for this query stem (x[1]['avg_total']/x[1]['doc_count'] -> this gives the normalization)
        neighbors = sorted(correlation[stem].items(),
                           key=lambda x: x[1]['avg_total']/x[1]['doc_count'], reverse=True)[:n_neighbors+20]

        for neighbor, score in neighbors:
            if neighbor not in query_terms and neighbor not in query_stems:  # avoid adding existing terms
                expansion_terms[neighbor] += score['avg_total']/score['doc_count']

    # 3. Combine original query with expansion terms
    sorted_expansion = sorted(expansion_terms.items(),
                              key=lambda x: x[1], reverse=True)


    expanded_query = [term for term, _ in sorted_expansion]

    # get unique set of words
    unique_stems = set(expanded_query)
    unique_stems_list = list(unique_stems)

    # get complete words
    refined_expanded_query = []
    for stem in unique_stems_list:
        full_word = complete_word(stem, all_tokens)
        if full_word != 'None' and is_real_word(full_word):
            refined_expanded_query.append(full_word)

    return ' '.join(query_terms + refined_expanded_query[:n_neighbors])


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

        expanded_query = metric_cluster_main(remove_stopwords_nltk(query), results)
        print("Expanded query terms:", expanded_query)

        qe_results = search(server=solr, query=expanded_query)

        print("QE Results:")
        # Print each result's fields
        for result in qe_results:
            print(result["title"])
