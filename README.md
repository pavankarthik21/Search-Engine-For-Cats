## Search Engine for Cats ![playful-kitten-peeking-over-edge-with-curious-expression-png](https://github.com/user-attachments/assets/d9075100-3a50-4104-b117-455822589b9a)


A modular, topic-focused search engine that crawls, indexes, and ranks web pages related to cats—covering everything from adoption and health to behavior and care. Built for the CS 6322 Information Retrieval course at The University of Texas at Dallas, this project leverages advanced IR techniques to deliver relevant, high-quality cat-related content.

---

### Features

- **Web-Scale Crawling:**
Crawls ~135,000 cat-related web pages using Apache Nutch, starting from 200 carefully curated seed URLs. De-duplication and incremental crawling ensure efficiency and freshness.
- **Advanced Indexing \& Ranking:**
Indexes content with Apache Solr. Supports multiple relevance models:
    - Vector Space (TF-IDF)
    - PageRank \& HITS (link analysis)
    - Combined vector and link-based ranking
- **Clustering:**
Implements both flat (KMeans, K=11) and agglomerative (hierarchical, ~10 clusters) document clustering using TF-IDF and SVD for dimensionality reduction. Clusters improve semantic grouping and search result diversity.
- **Query Expansion:**
Enhances user queries using the Rocchio algorithm and clustering-based expansion (association, metric, scalar), improving recall and handling ambiguous or misspelled queries.
- **Interactive Web UI:**
Angular frontend with Flask backend. Users can:
    - Enter queries and select ranking/clustering/expansion modes
    - Compare results from the custom engine, Google, and Bing side-by-side
    - Explore clustered results for deeper topic discovery

---

### Architecture

- **Crawler:** Apache Nutch
- **Indexer:** Apache Solr
- **Backend:** Python Flask
- **Frontend:** Angular
- **Clustering \& Ranking:** Python (scikit-learn, NetworkX)

---

### Example Use Cases

- Find adoption resources:
_"adopting a rescue cat"_ returns focused, authoritative results from shelters and adoption sites.
- Explore cat health:
_"common cat illnesses"_ surfaces educational and veterinary resources.
- Discover breeds or fun facts:
_"Sphynx cat"_ or _"funny cat memes"_ retrieves topical, clustered content.

---

### Key Learnings

- Switching from .NET to open-source tools (Nutch + Solr) vastly improved scalability and performance.
- Combining vector, link, and cluster-based models yields more accurate and diverse results than single-model approaches.
- Query expansion and clustering are essential for handling ambiguous or broad queries in a niche domain.

---

### Future Work

- Add more sophisticated query expansion and feedback mechanisms.
- Experiment with additional clustering methods.
- Scale to even larger datasets and improve UI/UX.

---

### Team

- Avaneesh Ramaseshan Baskaraswaminathan (UI, Integration)
- Venkata Subbaiah Pavan Karthik Navuluru (Indexing, Relevance Models)
- Dhanyan Muralidharan (Clustering)
- Varsha Viswanathan (Crawling)
- Sharon T Alexander (Query Expansion)

---

**For cat lovers, researchers, and anyone seeking credible, focused information about our feline friends!**
[See the full technical report for implementation details.](https://github.com/sharona1ex/Search-Engine-for-Cats/blob/a9103c4f5559f4c55fde4de21c7bdc096160fadb/CS%206322_IR%20report_Team%202_Search%20engine%20for%20Cats.pdf)[^1]

[^1]: CS-6322_IR-report_Team-2_Search-engine-for-Cats.pdf

<div style="text-align: center">⁂</div>

[^1]: CS-6322_IR-report_Team-2_Search-engine-for-Cats.pdf

