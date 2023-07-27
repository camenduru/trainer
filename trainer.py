import gradio as gr
from gradio import strings

from dreambooth import Dreambooth
from text_to_image import TextToImage
from textual_inversion import TextualInversion
# from lorad import LoraD
# from lora import Lora

trainer = gr.Blocks(title="Trainer")

def launch():
    # !git clone https://github.com/camenduru/sd-scripts
    # %cd /content/trainer/sd-scripts
    # !pip install -r /content/trainer/sd-scripts/requirements.txt
    strings.en["SHARE_LINK_MESSAGE"] = f"😊"
    with trainer:
        TextToImage.tab()
        Dreambooth.tab()
        TextualInversion.tab()
        # LoraD.tab()
        # Lora.tab()
    trainer.queue().launch(debug=True, share=True, inline=False)

if __name__ == "__main__":
    launch()
