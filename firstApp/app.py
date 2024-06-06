
import gradio as gr

def greet(name):
    return f"Hello {name} ! On est Online"

iface = gr.Interface(fn=greet, inputs="text", outputs="text")
iface.launch(server_name="0.0.0.0")