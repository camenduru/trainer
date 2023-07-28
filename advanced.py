import os, shutil
import gradio as gr
from gradio import strings

from dreambooth import Dreambooth
from text_to_image import TextToImage
from textual_inversion import TextualInversion
from lora_d_xl import LoraDXL
from lora_d import LoraD
# from lora import Lora

trainer = gr.Blocks(title="Trainer")

def upload_file(files):
    file_paths = [file.name for file in files]
    if not os.path.exists('/content/images'):
        os.mkdir('/content/images')
    for file_path in file_paths:
        shutil.copy(file_path, '/content/images/')
    return file_paths

def launch():
    # !git clone https://github.com/camenduru/sd-scripts
    # %cd /content/trainer/sd-scripts
    # !pip install -r /content/trainer/sd-scripts/requirements.txt
    strings.en["SHARE_LINK_MESSAGE"] = ""
    strings.en["BETA_INVITE"] = ""
    with trainer:
        with gr.Tab("Upload Images"):
            file_output = gr.File()
            upload_button = gr.UploadButton("Upload Images", file_types=["image"], file_count="multiple")
            upload_button.upload(upload_file, upload_button, file_output)
        TextToImage.tab()
        Dreambooth.tab()
        TextualInversion.tab()
        LoraD.tab()
        LoraDXL.tab()
        # Lora.tab()
    trainer.queue().launch(debug=True, share=True, inline=False)

if __name__ == "__main__":
    launch()
