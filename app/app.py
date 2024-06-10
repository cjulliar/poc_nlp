import gradio as gr
from dotenv import load_dotenv

from utils import (initialize_llm_and_embedding, 
                   preprocess_wikipedia, 
                   get_answer_llm,
                   preprocess_youtube)


retriever_store = {}
PREPROC = {
    "wiki": preprocess_wikipedia,
    "youtube": preprocess_youtube,
}

def define_user(user):
    global username
    username = user

def define_source(source_text, func):
    global source
    source = source_text
    retriever = PREPROC[func](source, embedding)
    retriever_store[source] = retriever

def query_llm(query, history):
    retriever = retriever_store[source]
    return get_answer_llm(username, retriever, query, llm) 

app = gr.Blocks()

with app:

    load_dotenv()
    llm, embedding = initialize_llm_and_embedding()

    # Description et session utilisateur
    gr.Markdown("# POC sur le concept du RAG")
    gr.Markdown("""
                Le RAG (Retrieval-Augmented Generation) est une technique d'IA qui combine la récupération d'informations 
                à partir de bases de données ou d'ensembles de documents avec la génération de texte pour fournir des réponses 
                précises et contextuelles. En utilisant le RAG, les systèmes d'IA peuvent consulter des sources externes pour 
                obtenir des informations pertinentes et les intégrer dans des réponses générées, améliorant ainsi la pertinence 
                et la précision des réponses fournies.
                """)
    user = gr.Textbox(label="Nom d'utilisateur")
    b1 = gr.Button("Définir l'utilisateur de la session")
    b1.click(define_user, inputs=user, outputs=None)

    # RAG wikipedia
    with gr.Tab("RAG Wikipedia"):
        text_page_name = gr.Textbox(label="Nom de la page Wikipedia")
        func = gr.Textbox(value="wiki", visible=False)
        b2 = gr.Button("Choisir cette page pour les recherches")
        b2.click(define_source, inputs=[text_page_name, func], outputs=None)
        chat = gr.ChatInterface(query_llm)

    # RAG YouTube
    with gr.Tab("RAG YouTube"):
        url = gr.Textbox(label="URL de la vidéo YouTube")
        func = gr.Textbox(value="youtube", visible=False)
        b3 = gr.Button("Choisir cette URL pour les recherches")
        b3.click(define_source, inputs=[url, func], outputs=None)
        chat = gr.ChatInterface(query_llm)


app.launch(server_name="0.0.0.0")