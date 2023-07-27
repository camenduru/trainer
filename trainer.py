import os, shutil
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
        with gr.Tab("Upload Images"):
            uploaded_files = gr.File(file_count="directory", file_types=["image"])
            if not os.path.exists('/content/images'):
                os.mkdir('/content/images')
            for uploaded_file in uploaded_files:
                shutil.copy(uploaded_file, '/content/images/')
        TextToImage.tab()
        Dreambooth.tab()
        TextualInversion.tab()
        # LoraD.tab()
        # Lora.tab()
    trainer.queue().launch(debug=True, share=True, inline=False)

if __name__ == "__main__":
    launch()
