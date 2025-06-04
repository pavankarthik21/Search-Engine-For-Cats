"""
Author: Sharon T Alexander
"""
import re
import collections
import heapq

import numpy as np
from nltk.corpus import stopwords
from nltk import PorterStemmer

import nltk
nltk.download('words')
from nltk.corpus import words
word_list = set(words.words())

import pysolr
import pprint
from solr_client import search
import pysolr


def is_real_word(word):
    return word in word_list


def complete_word(stem, words):
    for word in words:
        if word.startswith(stem):
            return word
    return 'None'

def remove_stopwords_nltk(query, language='english'):
    stop_words = set(stopwords.words(language))
    words = query.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

# returns a list of tokens
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

def get_top_word(word, association):
    associated_words = list(filter(lambda x: x[1]==word, association))
    sorted_associated_word = sorted(associated_words, key=lambda x: x[-1], reverse=True)
    return sorted_associated_word

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

def get_expanded_query(query, query_stems, association, all_tokens, k=2):
    n = len(query_stems)
    ex_qry = []
    for word in query_stems:
        word_list = get_top_word(word, association)
        # select the first two words from the length the query_stem index
        ex_qry.extend(extract_words(word_list[n:], all_tokens, exclude_words=ex_qry[:] + query_stems[:], k=k))


    # complete the stems
    expanded_query = ''
    for word in ex_qry:
        if not is_real_word(word):
            expanded_query += ' ' + complete_word(word, all_tokens)
        else:
            expanded_query += ' ' + word

    expanded_query = query + expanded_query

    return expanded_query



def build_association(id_token_map, vocab, query):
    association_list = []
    for i, voc in enumerate(vocab):
        for word in query.split(' '):
            c1, c2, c3 = 0, 0, 0
            for doc_id, tokens_this_doc in id_token_map.items():
                count0 = tokens_this_doc.count(voc)
                count1 = tokens_this_doc.count(word)
                c1 += count0 * count1
                c2 += count0 * count0
                c3 += count1 * count1
            c1 /= (c1 + c2 + c3)
            if c1 != 0:
                association_list.append((voc, word, c1))

    return association_list


def association_main(query, solr_results):
    results = []
    collected_result = {}
    for result in solr_results:
        if 'title' in result:
            if result['title'] not in collected_result:
                collected_result[result['title']] = 1
                results.append(result)

    print("QE V3")

    # re-assign
    solr_results = results

    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    all_tokens = set()
    # query = 'olympic medal'
    # solr = pysolr.Solr('http://localhost:8983/solr/nutch/', always_commit=True, timeout=10)
    # results = get_results_from_solr(query, solr)
    tokens = []
    token_counts = {}
    tokens_map = {}
    # tokens_map = collections.OrderedDict()
    document_ids = []

    for result in solr_results:
        if 'title' in result and 'content' in result and 'digest' in result:
            tokens_this_document_temp = tokenize_doc(result['title'] + ' ' + result['content'], stop_words)
            tokens_this_document = []
            for t in tokens_this_document_temp:
                tokens_this_document.append(stemmer.stem(t))
                all_tokens.add(t)
            tokens_map[result['digest']] = tokens_this_document
            tokens.append(tokens_this_document)

    vocab = set([token for tokens_this_doc in tokens for token in tokens_this_doc])
    query_terms = query.lower().split()
    query_stems = list(set([stemmer.stem(t) for t in query_terms]))
    stemmed_query = ' '.join(query_stems)
    association_list = build_association(tokens_map, vocab, stemmed_query)

    expanded_query = get_expanded_query(query, query_stems, association_list, all_tokens)

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

        expanded_query = association_main(remove_stopwords_nltk(query), results)
        print("Expanded query terms:", expanded_query)

        qe_results = search(server=solr, query=expanded_query)

        print("QE Results:")
        # Print each result's fields
        for result in qe_results:
            print(result["title"])