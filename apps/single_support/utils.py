import os
import re
import pymupdf
import pinecone

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
from langchain_pinecone import PineconeVectorStore

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# def extract_text_from_pdf(pdf_path):
#     doc = pymupdf.open(pdf_path)
#     text = ""
#     for page_num in range(len(doc)):
#         page = doc.load_page(page_num)
#         text += page.get_text()
#     return text
def extract_text_from_pdf(pdf_path):
    try:
        doc = pymupdf.open(pdf_path)
        text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text()
        return text
    except Exception as e:
        print(f"Error extracting text from PDF file {pdf_path}: {e}")
        return ""
    
def extract_text_from_docx(doc_path):
    try:
        doc = Document(doc_path)
        full_text = []
        for paragraph in doc.paragraphs:
            full_text.append(paragraph.text)
        return "\n".join(full_text)
    except Exception as e:
        print(f"Error extracting text from DOCX file {doc_path}: {e}")
        return ""

def load_documents_from_directory(directory_path):
    documents = []
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if filename.endswith(".pdf"):
            documents.append(extract_text_from_pdf(file_path))
        elif filename.endswith(".docx"):
            documents.append(extract_text_from_docx(file_path))
    return documents

def initialize_llm_and_embedding_and_pinecone():
    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("DEPLOYMENT_NAME_LLM"),
        openai_api_version="2023-06-01-preview",
        model_version="0301",
    )

    embedding = AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("DEPLOYMENT_NAME_EMBEDDING"),
        openai_api_version="2023-05-15",
    )

    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_client = pinecone.Pinecone(
        api_key=pinecone_api_key
    )

    return llm, embedding, pinecone_client

def preprocess_wikipedia(page_name, embedding):
    docs = WikipediaLoader(query=page_name, load_max_docs=1, doc_content_chars_max=10000).load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    doc_splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(doc_splits, embedding)
    retriever = vectorstore.as_retriever()
    return retriever

def preprocess_youtube(url, embedding):
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

    # Create vectorstore and retriever
    vectorstore = Chroma.from_documents(doc_splits, embedding)
    retriever = vectorstore.as_retriever()
    return retriever

def preprocess_pdf(pdf_path, embedding):
    pdf_text = extract_text_from_pdf(pdf_path)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    texts = text_splitter.split_text(pdf_text)
    docs = [Document(page_content=t) for t in texts]
    doc_splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(doc_splits, embedding)
    retriever = vectorstore.as_retriever()
    return retriever

def preprocess_corpus(directory_path, embedding):
    corpus = load_documents_from_directory(directory_path)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    doc_splits = []
    for text in corpus:
        texts = text_splitter.split_text(text)
        docs = [Document(page_content=t) for t in texts]
        doc_splits.extend(text_splitter.split_documents(docs))
    index_name = "corpus"
    pinecone = PineconeVectorStore.from_documents(doc_splits,embedding,index_name)
    retriever = pinecone.as_retriever()
    return retriever

def create_prompts_and_chains(llm, retriever):
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

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

    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
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
    response = conversational_rag_chain.invoke(
        {"input": query},
        config={"configurable": {"session_id": username}}
    )
    return response["answer"]

def get_answer_llm(username, retriever, query, llm):
    conversational_rag_chain = create_prompts_and_chains(llm, retriever)
    answer = execute_rag_chain(conversational_rag_chain, username, query)
    return answer
