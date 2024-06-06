import gradio as gr
from dotenv import load_dotenv

from utils import rag_wikipedia

def define_page(user, page):
    global username
    global page_name
    username = user
    page_name = page

def query_rag(query):
    return rag_wikipedia(username, page_name, query) 

app = gr.Blocks()

with app:

    load_dotenv()

    text_username = gr.Textbox()
    text_page_name = gr.Textbox()
    b1 = gr.Button("Choisir cette page pour les recherches")

    text_query = gr.Textbox()
    answer = gr.Textbox()
    b2 = gr.Button("Soumettre votre question")

    b1.click(define_page, inputs=[text_username, text_page_name], outputs=None)
    b2.click(query_rag, inputs=text_query, outputs=answer)


app.launch(server_name="0.0.0.0")