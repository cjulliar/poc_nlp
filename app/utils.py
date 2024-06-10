import os
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import WikipediaLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def initialize_llm_and_embedding():
    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("DEPLOYMENT_NAME_LLM"),
        openai_api_version="2023-06-01-preview",
        model_version="0301",
    )

    embedding = AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("DEPLOYMENT_NAME_EMBEDDING"),
        openai_api_version="2023-05-15",
    )
    return llm, embedding

def preprocess_wikipedia(page_name, embedding):
    docs = WikipediaLoader(query=page_name, load_max_docs=1, doc_content_chars_max=10000).load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    doc_splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(doc_splits, embedding)
    retriever = vectorstore.as_retriever()
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

def answer_wikipedia(username, retriever, query, llm):
    conversational_rag_chain = create_prompts_and_chains(llm, retriever)
    answer = execute_rag_chain(conversational_rag_chain, username, query)
    return answer
