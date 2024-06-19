import os
import re

import chromadb
import pymupdf
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import WikipediaLoader, YoutubeLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter

store = {}

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """
    Retourne l'historique des messages avec un Chat d'une session donnée.
    Si il n'existe pas d'historique pour cette session, en crée un.
    """
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def delete_session_history(session_id):
    """Supprime l'historique des messages avec Chat d'une session donnée."""
    if session_id in store:
        store.pop(session_id)


def initialize_llm_embedding_client():
    """
    Crée et retourne des instances de modèle Chat et Embedding à partir d'Azure OpenAI,
    et un client éphémère d'une base de données Chroma.
    """
    client = chromadb.Client()
    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("DEPLOYMENT_NAME_LLM"),
        openai_api_version="2023-06-01-preview",
        model_version="0301",
    )
    embedding = AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("DEPLOYMENT_NAME_EMBEDDING"),
        openai_api_version="2023-05-15",
    )
    return llm, embedding, client


def extract_text_from_pdf(pdf):
    """Extrait le texte d'un pdf et le retourne."""
    doc = pymupdf.open(pdf)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text


def extract_text_from_docx(docx):
    """Extrait le texte d'un docx et le retourne."""
    try:
        doc = Document(docx)
        full_text = []
        for paragraph in doc.paragraphs:
            full_text.append(paragraph.text)
        return "\n".join(full_text)
    except Exception as e:
        print(f"Error extracting text from DOCX file {docx}: {e}")
        return ""


def preprocess_wiki(page_name):
    """
    Récupère les données d'une page wikipedia et les retourne
    sous la forme d'une liste de plusieurs documents
    """
    docs = WikipediaLoader(query=page_name, load_max_docs=1, doc_content_chars_max=10000).load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    doc_splits = text_splitter.split_documents(docs)
    return doc_splits


def preprocess_youtube(url):
    """
    Récupère le transcript d'une vidéo youtube et le retourne 
    sous la forme d'une liste de plusieurs documents
    """ 
    # Extract video id
    id_regex = re.compile("v=(.+)$")
    matches = re.search(id_regex, url)

    if matches:
        id_ = matches.groups()[0]
    else:
        raise ValueError("No id found in the given URL")

    # Load transcript
    transcript_loader = YoutubeLoader(video_id=id_)
    transcription = transcript_loader.load()[0].page_content

    # Split transcript
    text_documents = [Document(page_content=transcription)]
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1_000, chunk_overlap=200
    )
    doc_splits = text_splitter.split_documents(text_documents)
    return doc_splits


def preprocess_pdf(pdf):
    """
    Récupère le texte d'un pdf et le retourne 
    sous la forme d'une liste de plusieurs documents
    """
    pdf_text = extract_text_from_pdf(pdf)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    texts = text_splitter.split_text(pdf_text)
    docs = [Document(page_content=t) for t in texts]
    doc_splits = text_splitter.split_documents(docs)
    return doc_splits


def preprocess_textfiles(files):
    """
    Récupère le texte de plusieurs fichiers pdf ou docx et les 
    retourne sous la forme d'une liste de plusieurs documents
    """
    docs = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    for file in files:
        if file.endswith(".pdf"):
            print(True)
            file_text = extract_text_from_pdf(file)
        elif file.endswith(".docx"):
            file_text = extract_text_from_docx(file)
        texts = text_splitter.split_text(file_text)
        docs.extend(Document(page_content=t) for t in texts)
    doc_splits = text_splitter.split_documents(docs)
    return doc_splits


def create_vectorstore_and_retriever(docs, embedding, client, username):
    """
    Crée un vectorstore de documents qui est enregistré au sein
    d'une collection du client Chroma, et retourne un retriever
    de ce vectorstore.
    """
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embedding,
        client=client,
        collection_name=f"{username}_collection",
    )
    retriever = vectorstore.as_retriever()
    return retriever


def create_prompts_and_chains(llm, retriever):
    """
    Crée et retourne une rag chain où le retriever et le llm ont accès
    à l'historique des messages du Chat.
    """
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    return conversational_rag_chain


def execute_rag_chain(conversational_rag_chain, username, query):
    """
    Transmets une query et une session utilisateur 
    à une rag chain et retourne la réponse.
    """
    response = conversational_rag_chain.invoke(
        {"input": query},
        config={"configurable": {"session_id": username}}
    )
    for doc in response["context"]:
        print(doc)
    return response["answer"]


def get_answer_llm(username, retriever, query, llm):
    """Crée une rag chain, lui transmets une query et retourne la réponse."""
    conversational_rag_chain = create_prompts_and_chains(llm, retriever)
    answer = execute_rag_chain(conversational_rag_chain, username, query)
    return answer