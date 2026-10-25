import gradio as gr
from agent import agent              # the function you just wrote

def chat(message, history):          # ① Gradio fills these two in for you
    return agent(message)

gr.ChatInterface(fn=chat, title="🛍️ Smart Shop Assistant").launch(share=True)  # ②