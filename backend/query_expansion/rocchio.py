import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pysolr

def search(server, query):
    # Perform a search query (e.g., search for documents containing 'cheese')
    q = query
    search_string = f'title:({q}) AND content:({q})'
    print(search_string)
    output = server.search(search_string, rows=10, fl='id,title,content,digest')
    results = []
    for result in output:
        results.append(result)
    return results


class RocchioExpander:
    def __init__(self, alpha=1.0, beta=0.8, gamma=0.1):
        self.alpha = alpha  # Original query weight
        self.beta = beta  # Relevant docs weight
        self.gamma = gamma  # Non-relevant docs weight
        self.vectorizer = TfidfVectorizer()
        self.vocab = None

    def fit_transform(self, documents):
        tfidf_matrix = self.vectorizer.fit_transform(documents)
        self.vocab = self.vectorizer.get_feature_names_out()
        return tfidf_matrix

    def expand_query(self, original_query, relevant_docs, non_relevant_docs=[]):
        # Convert documents to TF-IDF vectors
        query_vec = self.vectorizer.transform([original_query]).toarray()[0]
        relevant_matrix = self.vectorizer.transform(relevant_docs).toarray()
        non_relevant_matrix = self.vectorizer.transform(non_relevant_docs).toarray()

        # Calculate centroids
        rel_centroid = np.mean(relevant_matrix, axis=0) if len(relevant_docs) > 0 else 0
        non_rel_centroid = np.mean(non_relevant_matrix, axis=0) if len(non_relevant_docs) > 0 else 0

        # Apply Rocchio formula [1][2][5]
        new_query = (
                self.alpha * query_vec +
                self.beta * rel_centroid -
                self.gamma * non_rel_centroid
        )

        # Remove negative weights (common practice) [3]
        new_query = np.clip(new_query, a_min=0, a_max=None)

        return self._vec_to_terms(new_query)

    def _vec_to_terms(self, vector, top_n=5):
        # Get top terms with highest weights
        sorted_indices = np.argsort(vector)[::-1]
        return [(self.vocab[i], vector[i]) for i in sorted_indices if vector[i] > 0][:top_n]


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
            print(f'{idx}: {result}')

        # get content by combining the title and content part
        documents = [result['title'] + ' ' + result['content'] for result in results]

        rocchio = RocchioExpander()
        rocchio.fit_transform(documents)

        relevant_indices = list(map(int, input("Enter Index of Relevant Documents (For eg. 1,4,5):").split(",")))
        non_relevant_indices = list(map(int, input("Enter Index of Non Relevant Documents (For eg. 3,8):").split(",")))

        relevant_docs = [documents[idx] for idx in relevant_indices]  # User feedback
        non_relevant_docs = [documents[idx] for idx in non_relevant_indices]  # User feedback

        expanded_terms = rocchio.expand_query(original_query, relevant_docs, non_relevant_docs)
        expanded_query = " ".join([term for term, weight in expanded_terms])
        print("Expanded query terms:", expanded_query)

        qe_results = search(server=solr, query=expanded_query)

        print("QE Results:")
        # Print each result's fields
        for result in qe_results:
            print(result)
