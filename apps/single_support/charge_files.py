import os
from langchain_chroma import Chroma
from langchain_community.document_loaders import LocalFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def preprocess_documents(document_paths, embedding):
    docs = []
    for path in document_paths:
        loader = LocalFileLoader(file_path=path)
        docs.extend(loader.load())
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    doc_splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(doc_splits, embedding)
    return vectorstore

# Liste des chemins des documents à utiliser pour l'entraînement
document_paths = ["path/to/doc1.txt", "path/to/doc2.txt", "path/to/doc3.txt"]

# Initialiser l'embedding et charger les documents
llm, embedding = initialize_llm_and_embedding()
global_vectorstore = preprocess_documents(document_paths, embedding)
