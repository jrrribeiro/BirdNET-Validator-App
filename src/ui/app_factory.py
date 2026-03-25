import gradio as gr


def create_app() -> gr.Blocks:
    with gr.Blocks(title="BirdNET Validator HF") as demo:
        gr.Markdown("# BirdNET Validator HF")
        gr.Markdown("Sprint 0 scaffold running.")

    return demo
