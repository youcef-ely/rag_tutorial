import os
import logging
import pretty_errors
from tqdm.notebook import tqdm
from configs.configs import Config

from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface import HuggingFaceEmbeddings  
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter



#Initializing the global variables
conversation_retrieval_chain = None
chat_history = []
llm_hub = None
embeddings = HuggingFaceEmbeddings(model_kwargs={'device': Config.DEVICE})
persist_directory = '../data/chroma_langchain_db'



def process_document(file_path):    
    global embeddings, conversation_retrieval_chain, llm_hub

    loader = PyPDFLoader(file_path = file_path)
    document = loader.load()
    
    # Text splitting
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=1000,
        chunk_overlap=200
    )

    splits = text_splitter.split_documents(document)
    logging.info("Text splitted successfully!")
    
        
    # Create the vector store with precomputed embeddings
    db = Chroma.from_documents(
        collection_name=Config.COLLECTION_NAME,
        documents=splits,
        embedding=embeddings,
        persist_directory=persist_directory
    )

    db.persist()
    logging.info("Vector store created successfully!")

    


def process_prompt(prompt):
    global conversation_retrieval_chain, chat_history, embeddings

    db = Chroma(
    persist_directory=persist_directory,
    embedding_function=embeddings)

    retriever = db.as_retriever(search_type="mmr", search_kwargs={'k': 6, 'lambda_mult': 0.25})

    llm_hub = HuggingFaceEndpoint(repo_id=Config.GENERATOR_MODEL_NAME, 
                                 temperature=0.1, max_new_tokens=600)
    
    conversation_retrieval_chain = RetrievalQA.from_chain_type(
        llm=llm_hub,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,
        input_key = "question"
    )
    
    # Query the model
    output = conversation_retrieval_chain({"question": prompt, "chat_history": chat_history})
    answer = output["result"]
    
    # Update the chat history
    chat_history.append((prompt, answer))
    
    # Return the model's response
    return answer






