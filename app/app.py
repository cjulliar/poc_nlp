import gradio as gr
from dotenv import load_dotenv

from utils import initialize_llm_and_embedding, preprocess_wikipedia, answer_wikipedia

retriever_store = {}

def define_user(user):
    global username
    username = user

def define_page(page):
    global page_name
    page_name = page
    retriever = preprocess_wikipedia(page_name, embedding)
    retriever_store[page_name] = retriever

def query_rag(query, history):
    retriever = retriever_store[page_name]
    return answer_wikipedia(username, retriever, query, llm) 

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
        b2 = gr.Button("Choisir cette page pour les recherches")
        chat = gr.ChatInterface(query_rag)
        b2.click(define_page, inputs=text_page_name, outputs=None)

    # RAG autre
    with gr.Tab("RAG sur un autre sujet"):
        pass


app.launch(server_name="0.0.0.0")