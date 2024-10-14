import os
import sys
import time
import torch
import logging
import pandas as pd
from typing import Literal

sys.path.append('../')
from rag.config import Config
from llama_parse import LlamaParse
from llama_index.core import Settings
from qdrant_client import QdrantClient
from llama_index.llms.groq import Groq
from langchain.vectorstores import Chroma
from llama_index.core import StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.core.indices.vector_store.base import VectorStoreIndex

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

qdrant_client = QdrantClient(
    url = Config.QDRANT_URL,
    prefer_grpc = True,
    api_key = Config.QDRANT_API_KEY
)

llm = Groq(model = "llama3-70b-8192", api_key = Config.GROQ_API_KEY)
Settings.llm = llm

embed_model = FastEmbedEmbedding(model_name = "BAAI/bge-small-en-v1.5")
Settings.embed_model = embed_model

vector_store = QdrantVectorStore(client = qdrant_client, collection_name = Config.COLLECTION_NAME)
db_index = VectorStoreIndex.from_vector_store(vector_store = vector_store)




def parse_dataset(dataset_name: Literal['sustainability', 'christmas']) -> dict:
    if not Config.LLAMA_API_KEY:
        logging.error("LLAMA_API_KEY not found in environment variables.")
        return {"error": "Missing API key"}
    
    parser = LlamaParse(api_key = Config.LLAMA_API_KEY)

    try:
        if dataset_name not in Config.DATA_PATHS:
            raise ValueError(f"{dataset_name} dataset is not supported")

        parser.parsing_instruction = Config.INSTRUCTIONS[dataset_name]
        doc = parser.load_data(Config.DATA_PATHS[dataset_name])
        logging.info(f"Successfully parsed dataset: {dataset_name}")
        return doc

    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        return {"error": "File not found"}

    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return {"error": str(e)}


def embedding_doc(parsed_doc):
    try:
        storage_context = StorageContext.from_defaults(vector_store = vector_store)
        VectorStoreIndex.from_documents(parsed_doc, storage_context = storage_context)
        logging.info("Document embedded and indexed successfully.")
    except Exception as e:
        logging.error(f"Failed to embed and store document: {e}")
        raise e



def retrieval_and_generate(query: str) -> dict:
    try:
        query_engine = db_index.as_query_engine()
        start_time = time.time()
        response = query_engine.query(query)
        elapsed_time = time.time() - start_time

        logging.info(f"Query executed in {elapsed_time:.2f} seconds.")
        logging.info(f"Query: {query}")
        logging.info(f"Response: {response.response}")

        return response.response

    except Exception as e:
        logging.error(f"Query failed: {e}")
        return {"error": str(e)}
