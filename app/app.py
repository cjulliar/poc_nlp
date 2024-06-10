import gradio as gr
from dotenv import load_dotenv

from utils import rag_wikipedia

def define_page(user, page):
    global username
    global page_name
    username = user
    page_name = page

def change_page():
    pass

def query_rag(query, history):
    return rag_wikipedia(username, page_name, query) 

app = gr.Blocks()

with app:

    load_dotenv()

    gr.Markdown("POC sur le concept du RAG.")

    with gr.Tab("RAG Wikipedia"):

        text_username = gr.Textbox()
        text_page_name = gr.Textbox()
        b1 = gr.Button("Choisir cette page pour les recherches")

        chat = gr.ChatInterface(query_rag)

        b1.click(define_page, inputs=[text_username, text_page_name], outputs=None)

    with gr.Tab("RAG sur un autre sujet"):
        pass


app.launch(server_name="0.0.0.0")