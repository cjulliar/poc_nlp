import gradio as gr
from dotenv import load_dotenv

from utils import (create_vectorstore_and_retriever,
                   delete_session_history,
                   get_answer_llm,
                   initialize_llm_embedding_client,
                   preprocess_textfiles,
                   preprocess_wiki,
                   preprocess_youtube)


PREPROC = {
    "textfiles": preprocess_textfiles,
    "wiki": preprocess_wiki,
    "youtube": preprocess_youtube,
}

def define_user(user):
    global username
    username = user

def create_retriever(source, func):
    global retriever
    docs = PREPROC[func](source)
    retriever = create_vectorstore_and_retriever(docs, embedding, client, username)

def query_llm(query, history):
    return get_answer_llm(username, retriever, query, llm)

app = gr.Blocks()

with app:

    load_dotenv()
    llm, embedding, client = initialize_llm_embedding_client()

    # Description et session utilisateur
    gr.Markdown("# POC sur le concept d'un RAG multi-support")
    gr.Markdown("""
                Le RAG (Retrieval-Augmented Generation) est une technique d'IA qui combine la récupération d'informations 
                à partir de bases de données ou d'ensembles de documents avec la génération de texte pour fournir des réponses 
                précises et contextuelles. En utilisant le RAG, les systèmes d'IA peuvent consulter des sources externes pour 
                obtenir des informations pertinentes et les intégrer dans des réponses générées, améliorant ainsi la pertinence 
                et la précision des réponses fournies. Ce POC vise à démontrer l'intérêt d'un RAG pour prendre en compte un
                contexte issu de différentes sources afin de fournir une réponse à un utilisateur.
                """)
    user = gr.Textbox(label="Nom d'utilisateur")
    b1 = gr.Button("Définir l'utilisateur de la session")
    b2 = gr.Button("Supprimer la session de l'utilisateur")
    b1.click(define_user, inputs=user, outputs=None)
    b2.click(delete_session_history, inputs=user, outputs=None)

    gr.Markdown("## Supports disponibles :")
    # Support wikipedia
    with gr.Tab("Ajouter une page Wikipedia"):
        text_page_name = gr.Textbox(label="Nom de la page Wikipedia")
        func = gr.Textbox(value="wiki", visible=False)
        b3 = gr.Button("Sélectionner cette page Wikipedia")
        b3.click(create_retriever, inputs=[text_page_name, func], outputs=None)

    # Support youtube
    with gr.Tab("Ajouter une vidéo youtube"):
        url = gr.Textbox(label="URL de la vidéo YouTube")
        func = gr.Textbox(value="youtube", visible=False)
        b4 = gr.Button("Choisir cette vidéo youtube")
        b4.click(create_retriever, inputs=[url, func], outputs=None)

    # Support pdf et docx
    with gr.Tab("Ajouter des documents pdf/docx"):
        files = gr.File(label="Charger un ou plusieurs fichiers pdf/docx", file_count="multiple", file_types=[".pdf", ".docx"])
        func = gr.Textbox(value="textfiles", visible=False)
        b4 = gr.Button("Sélectionner ces fichiers")
        b4.click(create_retriever, inputs=[files, func], outputs=None)

    chat = gr.ChatInterface(query_llm, title="Interface de Chatbot", fill_height=False)

app.launch(server_name="127.0.0.1", server_port=8000)