from pymongo import MongoClient
from pymongo.operations import SearchIndexModel

import config

client = MongoClient(config.MONGODB_ATLAS_CLUSTER_URI)

DB_NAME = "team-10"
COLLECTION_NAME = "rag"
ATLAS_VECTOR_SEARCH_INDEX_NAME = "team_10_rag"

MONGODB_COLLECTION = client[DB_NAME][COLLECTION_NAME]

MONGODB_COLLECTION.update_search_index(
    definition={
        "fields": [
            {
                "numDimensions": 1536,
                "path": "embedding",
                "similarity": "cosine",
                "type": "vector"
            }
        ]
    },
    name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
)

from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import os

# os.environ["OPENAI_API_KEY"] = config.OPENAI_API_KEY

from resources import resources

docs = []
for resource in resources.RESOURCES:
    loader = PyPDFLoader(resource["path"]) if resource["type"] == "pdf" else \
            WebBaseLoader(resource["url"]) # if resource["type"] == "webpage"
    print(loader)
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs.extend(text_splitter.split_documents(data))

from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_community.embeddings import OpenAIEmbeddings

embedding = OpenAIEmbeddings()

vector_store = MongoDBAtlasVectorSearch.from_documents(
    documents=docs,
    embedding=embedding,
    collection=MONGODB_COLLECTION,
    index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
)
retriever = vector_store.as_retriever()