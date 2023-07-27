import gradio as gr
from gradio import strings

from dreambooth import Dreambooth
from text_to_image import TextToImage
from textual_inversion import TextualInversion
from lora import Lora

trainer = gr.Blocks(title="Trainer")

def launch():
    strings.en["SHARE_LINK_MESSAGE"] = f"😊"
    with trainer:
        TextToImage.tab()
        Dreambooth.tab()
        TextualInversion.tab()
        Lora.tab()
    trainer.queue().launch(debug=True, share=True, inline=False)

if __name__ == "__main__":
    launch()
