"""
Author: Sharon T Alexander
"""

import pysolr



def search(server, query):
    # Perform a search query (e.g., search for documents containing 'cheese')
    q = query
    search_string = f'title:({q}) AND content:({q})'
    print(search_string)
    output = server.search(search_string, search_handler="/select", rows=20, fl='id,title,content,digest,url')
    # print(output.hits)
    # results = []
    # collected_result = {}
    # for result in output:
    #     if result['title'] not in collected_result:
    #         collected_result[result['title']] = 1
    #         results.append(result)
    return output


if __name__ == '__main__':
    # Connect to your Solr core (replace with your actual Solr URL and core name)
    solr = pysolr.Solr('http://localhost:8983/solr/nutch', timeout=10, always_commit=True)

    results = search(solr, 'wild cats')

    # Print the number of results found
    print(f"Saw {len(results)} result(s).")

    # Print each result's fields
    for result in results:
        print(result)
