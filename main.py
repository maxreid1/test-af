import azure.functions as func
import datetime
import json
import logging
from azure.search.documents import SearchClient, VectorizedQuery
from azure.core.credentials import AzureKeyCredential
import openai
import os
from azure.search.documents.models import (
    QueryAnswerType,
    QueryCaptionType,
    QueryType,
    VectorizedQuery
)

app = func.FunctionApp()

def generate_embeddings(text):
    if not text:
        return func.HttpResponse(
            "Please provide text in the query string.",
            status_code=400
        )

    openai.api_key = os.getenv("OPENAI_API_KEY")
    response = openai.embeddings.create(
        input=text,
        model=os.getenv("EMBEDDING_MODEL")
    )
    embedding = response.data[0].embedding
    return embedding

@app.route(route="HttpExample2", auth_level=func.AuthLevel.ANONYMOUS)
def HttpExample2(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    name = req.params.get('name')
    if not name:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            name = req_body.get('name')

    if name:
        return func.HttpResponse(f"Hello, {name}. This HTTP triggered function executed successfully.")
    else:
        return func.HttpResponse(
             "This HTTP triggered function executed successfully. Pass a name in the query string or in the request body for a personalized response.",
             status_code=200
        )

@app.route(route="vector_similarity_search", auth_level=func.AuthLevel.ANONYMOUS)
def vector_similarity_search(req: func.HttpRequest) -> func.HttpResponse:
    search_service_endpoint = req.params.get('search_service_endpoint')
    index_name = req.params.get('index_name')
    query = req.params.get('query')
    k_nearest_neighbors = req.params.get('k_nearest_neighbors')
    search_column = req.params.get('search_column')
    use_hybrid_query = req.params.get('use_hybrid_query')
    vector_query = VectorizedQuery(vector=generate_embeddings(query), k_nearest_neighbors=k_nearest_neighbors, fields=search_column)

    if not (search_service_endpoint and index_name and query):
        return func.HttpResponse(
            "Please provide search_service_endpoint, index_name, and query in the query string.",
            status_code=400
        )

    search_client = SearchClient(
        endpoint=search_service_endpoint,
        index_name=index_name,
        credential=AzureKeyCredential(os.getenv("SEARCH_SERVICE_API_KEY"))
    )
    search_text = query if use_hybrid_query else None
    results = search_client.search(  
        search_text=search_text,  
        vector_queries=[vector_query]
    )
    return func.HttpResponse(json.dumps([result for result in results]), mimetype="application/json")

@app.route(route="vector_similarity_search_semantic_reranking", auth_level=func.AuthLevel.ANONYMOUS)
def vector_similarity_search_semantic_reranking(req: func.HttpRequest) -> func.HttpResponse:
    search_service_endpoint = req.params.get('search_service_endpoint')
    index_name = req.params.get('index_name')
    query = req.params.get('query')
    k_nearest_neighbors = req.params.get('k_nearest_neighbors')
    search_column = req.params.get('search_column')
    use_hybrid_query = req.params.get('use_hybrid_query')
    vector_query = VectorizedQuery(vector=generate_embeddings(query), k_nearest_neighbors=k_nearest_neighbors, fields=search_column)

    if not (search_service_endpoint and index_name and query):
        return func.HttpResponse(
            "Please provide search_service_endpoint, index_name, and query in the query string.",
            status_code=400
        )

    search_client = SearchClient(
        endpoint=search_service_endpoint,
        index_name=index_name,
        credential=AzureKeyCredential(os.getenv("SEARCH_SERVICE_API_KEY"))
    )
    results = search_client.search(
        search_text=query,
        vector_queries=[vector_query],
        query_type=QueryType.SEMANTIC,
        semantic_configuration_name="my-semantic-config",
        query_caption=QueryCaptionType.EXTRACTIVE,
        query_answer=QueryAnswerType.EXTRACTIVE,
        top=3,
    )

    response_data = {
        "semantic_answers": [],
        "results": []
    }

    semantic_answers = results.get_answers()
    for answer in semantic_answers:
        answer_data = {
            "text": answer.highlights if answer.highlights else answer.text,
            "score": answer.score
        }
        response_data["semantic_answers"].append(answer_data)

    for result in results:
        result_data = {
            "title": result['title'],
            "reranker_score": result['@search.reranker_score'],
            "url": result['url'],
            "caption": result["@search.captions"][0].highlights if result["@search.captions"] and result["@search.captions"][0].highlights else result["@search.captions"][0].text if result["@search.captions"] else None
        }
        response_data["results"].append(result_data)

    return func.HttpResponse(json.dumps(response_data), mimetype="application/json")
