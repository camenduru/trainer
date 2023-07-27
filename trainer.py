import os, shutil
import gradio as gr
from gradio import strings

from dreambooth import Dreambooth
from text_to_image import TextToImage
from textual_inversion import TextualInversion
# from lorad import LoraD
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
    strings.en["SHARE_LINK_MESSAGE"] = f"😊"
    with trainer:
        with gr.Tab("Upload Images"):
            # uploaded_files = gr.File(file_count="directory", file_types=["image"])
            # if not os.path.exists('/content/images'):
            #     os.mkdir('/content/images')
            # for uploaded_file in uploaded_files:
            #     shutil.copy(uploaded_file, '/content/images/')
            file_output = gr.File()
            upload_button = gr.UploadButton(file_types=["image"], file_count="multiple")
            upload_button.upload(upload_file, upload_button, file_output)
        TextToImage.tab()
        Dreambooth.tab()
        TextualInversion.tab()
        # LoraD.tab()
        # Lora.tab()
    trainer.queue().launch(debug=True, share=True, inline=False)

if __name__ == "__main__":
    launch()
