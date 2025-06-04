import flask
from flask_cors import CORS
import pysolr
import re
from flask import request, jsonify
import json
from clustering.utils import find_similar_documents, load_vectorizer_vectors_urls
from query_expansion.association_cluster import association_main
from query_expansion.metric_cluster import metric_cluster_main
from query_expansion.rocchio import search as rochhio_search
import query_expansion.solr_client as solr_client
from spellchecker import SpellChecker

spell = SpellChecker()
solr_url = "http://localhost:8983/solr/nutch/"
solr = pysolr.Solr(solr_url, always_commit=True, timeout=10)

app = flask.Flask(__name__)
CORS(app)
app.config["DEBUG"] = True


@app.route('/test', methods=['GET'])
def test_search(debug=False):
    res = solr.search("NOT content:*industry* AND content:*metal*", search_handler="/select", **{
                'wt': 'json',
                'rows': 10,
            })
    if debug:
        print(res.docs)
    return jsonify(res.docs)

@app.route('/index', methods=['GET'])
def get_query():
    if 'query' in request.args and 'type' in request.args:
        query = str(request.args['query'])
        type =  str(request.args['type'])
        
        total_results = 20
        if type == "association_qe" or type == "metric_qe" or type == "scalar_qe":
            total_results = 20

        solr_results = get_results_from_solr(get_solr_multi_query(query), total_results)
        api_resp = parse_solr_results(solr_results)
        print("API response", api_resp)
        if type == "page_rank":
            print("Page rank results")
            result = api_resp
        elif "clustering" in type:
            result = get_clustering_results(api_resp, type)
        elif type == "hits":
            result = get_hits_results(api_resp)
        elif type == "association_qe":
            solr_results = solr_client.search(solr, query)
            expanded_query = association_main(query, solr_results)
            # query = get_solr_multi_query(expanded_query)
            solr_res_after_qe = solr_client.search(solr, expanded_query)
            api_resp = parse_solr_results(solr_res_after_qe)
            result = api_resp
            result[0]['expanded_query'] = expanded_query
        elif type == "metric_qe":
            solr_results = solr_client.search(solr, query)
            expanded_query = metric_cluster_main(query, solr_results)
            # query = get_solr_multi_query(expanded_query)
            solr_res_after_qe = solr_client.search(solr, expanded_query)
            api_resp = parse_solr_results(solr_res_after_qe)
            result = api_resp
            result[0]['expanded_query'] = expanded_query
        elif type == "scalar_qe":
            solr_results = solr_client.search(solr, query)
            expanded_query = association_main(query, solr_results)
            # query = get_solr_multi_query(expanded_query)
            solr_res_after_qe = solr_client.search(solr, expanded_query)
            api_resp = parse_solr_results(solr_res_after_qe)
            result = api_resp
            result[0]['expanded_query'] = expanded_query
        elif type == "roc_qe":
            # # print("Received query: ", query)
            # expanded_query = calculate_term_weights(solr, solr_results, get_solr_multi_query(query) if len(query.split(' '))>1 else get_solr_query(query), query)
            # query = get_solr_multi_query(expanded_query)
            # solr_res_after_qe = get_results_from_solr(query, 20)
            # api_resp = parse_solr_results(solr_res_after_qe)
            # result = ""api_resp""
            # result[0]['expanded_query'] = expanded_query
            result = [{'title':"Rocchio is irrelevant without user feedback. Therefore not included here.",
                       'url':"",
                       'rank':1}]

        return jsonify(result)
    else:
        return "Error: No query or type provided"

@app.route('/cluster', methods=['GET'])
def search():
    query = request.args.get('query')
    if not query:
        return jsonify({'error': 'Query parameter is missing'}), 400

    # Load stored data
    vectorizer, document_vectors, urls = load_vectorizer_vectors_urls()

    # Find top 10 similar documents
    top_indices, top_similarities = find_similar_documents(query, vectorizer, document_vectors, top_n=10)
    #results = [{'url': urls[index], 'similarity': top_similarities[i]} for i, index in enumerate(top_indices)]
    results = [{'url': urls[index]} for i, index in enumerate(top_indices)]
    
    return jsonify(results)

def get_results_from_solr(query, no_of_results, debug=False):
    print(query)
    results = solr.search(query, search_handler="/select", **{
        "wt": "json",
        "rows": no_of_results
    })
    if debug:
        print(query, results)
    return results


def parse_solr_results(solr_results):
    if solr_results.hits == 0:
        return jsonify("query out of scope")
    else:
        api_resp = list()
        rank = 0
        for result in solr_results:
            rank += 1
            title = ""
            url = ""
            content = ""
            if 'title' in result:
                title = result['title']
                if 'url' in result:
                    url = result['url']
                if 'content' in result:
                    content = result['content']
                    meta_info = content[:200]
                    meta_info = meta_info.replace("\n", " ")
                    meta_info = " ".join(re.findall("[a-zA-Z]+", meta_info))
                link_json = {
                    "title": title,
                    "url": url,
                    "meta_info": meta_info,
                    "rank": rank
                }
                api_resp.append(link_json)
    return api_resp


def get_clustering_results(clust_inp, param_type):
    if param_type == "flat_clustering":
        f = open('clustering/clustering_result/clustering_f.txt')
        lines = f.readlines()
        f.close()
    elif param_type == "aglo_single_clustering":
        f = open('clustering/clustering_result/clustering_h_single.txt')
        lines = f.readlines()
        f.close()
    elif param_type == "aglo_complete_clustering":
        f = open('clustering/clustering_result/clustering_h_ward.txt')
        lines = f.readlines()
        f.close()

    cluster_map = {}
    for line in lines:
        line_split = line.split(",")
        if line_split[1] == "":
            line_split[1] = "99"
        cluster_map.update({line_split[0]: line_split[1]})

    for curr_resp in clust_inp:
        curr_url = curr_resp["url"]
        curr_cluster = cluster_map.get(curr_url, "99")
        curr_resp.update({"cluster": curr_cluster})
        curr_resp.update({"done": "False"})

    clust_resp = []
    curr_rank = 20
    for curr_resp in clust_inp:
        if curr_resp["done"] == "False":
            curr_cluster = curr_resp["cluster"]
            print("Current cluster")
            print(curr_cluster)
            curr_resp.update({"done": "True"})
            curr_resp.update({"rank": str(curr_rank)})
            curr_rank -= 1
            clust_resp.append({"title": curr_resp["title"], "url": curr_resp["url"],
                               "meta_info": curr_resp["meta_info"], "rank": curr_resp["rank"],
                               "cluster_no": curr_cluster})
            for remaining_resp in clust_inp:
                if remaining_resp["done"] == "False":
                    if remaining_resp["cluster"] == curr_cluster:
                        remaining_resp.update({"done": "True"})
                        remaining_resp.update({"rank": str(curr_rank)})
                        curr_rank -= 1
                        clust_resp.append({"title": remaining_resp["title"], "url": remaining_resp["url"],
                                           "meta_info": remaining_resp["meta_info"], "rank": remaining_resp["rank"],
                                           "cluster_no": curr_cluster})
    
    print("These are clustering results")
    print(clust_resp)
    
    return clust_resp


def get_hits_results(clust_inp):
    authority_score_file = open("Hits/authority_scores.txt", 'r').read()
    authority_score_dict = json.loads(authority_score_file)
    # print("Authority score dict", authority_score_dict)
    hits_resp = sorted(clust_inp, key=lambda x: authority_score_dict.get(x['url'], 0.0), reverse=True)
    # print(clust_resp)
    print("These are hits results")
    print(hits_resp)
    return hits_resp

def get_solr_query(query):
    return "text:"+query+""

def get_solr_multi_query(query, debug=False):
    query_terms = query.split()
    query = ""
    for i, term in enumerate(query_terms):
        query += "text:"
        # query += "*" + term + "*"
        query += term
        if i!=len(query_terms)-1:
            query += " OR "
    if debug: 
        print(query)
    return query


app.run(port='5000')
