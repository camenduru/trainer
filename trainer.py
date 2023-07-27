import os, time, torch
import gradio as gr
from subprocess import getoutput
from diffusers import StableDiffusionPipeline
from gradio import strings

from dreambooth import Dreambooth
from text_to_image import TextToImage
from textual_inversion import TextualInversion
from lora import Lora

def run_live(command):
  with os.popen(command) as pipe:
    for line in pipe:
      line = line.rstrip()
      print(line)
      yield line

def run_static(command):
    out = getoutput(f"{command}")
    print(out)
    return out

def timeout_test(second):
    start_time = time.time()
    while time.time() - start_time < int(second):
        pass
    msg = "🥳"
    return msg

pipe = None

def test_text_to_image(output_dir, load_model, prompt, negative_prompt, num_inference_steps, guidance_scale):
    global pipe
    if load_model:
        pipe = StableDiffusionPipeline.from_pretrained(output_dir, safety_checker=None, torch_dtype=torch.float16).to("cuda")
        pipe.enable_xformers_memory_efficient_attention()
    image = pipe(prompt, negative_prompt=negative_prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).images[0]
    return image

def test_dreambooth(output_dir, load_model, prompt, negative_prompt, num_inference_steps, guidance_scale):
    global pipe
    if load_model:
        pipe = StableDiffusionPipeline.from_pretrained(output_dir, safety_checker=None, torch_dtype=torch.float16).to("cuda")
        pipe.enable_xformers_memory_efficient_attention()
    image = pipe(prompt, negative_prompt=negative_prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).images[0]
    return image

def test_lora(model_dir, load_model, output_dir, prompt, negative_prompt, num_inference_steps, guidance_scale):
    global pipe
    if load_model:
        pipe = StableDiffusionPipeline.from_pretrained(model_dir, safety_checker=None, torch_dtype=torch.float16).to("cuda")
        pipe.enable_xformers_memory_efficient_attention()
        pipe.unet.load_attn_procs(output_dir)
    image = pipe(prompt, negative_prompt=negative_prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).images[0]
    return image

def clear_out_text():
    return ""

trainer = gr.Blocks(title="Trainer")

def launch():
    # colab_url = os.getenv('colab_url')
    # strings.en["SHARE_LINK_MESSAGE"] = f"WebUI Colab URL: {colab_url}"
    strings.en["SHARE_LINK_MESSAGE"] = f"😊"
    with trainer:
        TextToImage.tab()
        Dreambooth.tab()
        TextualInversion.tab()
        Lora.tab()
    trainer.queue().launch(debug=True, share=True, inline=False)

if __name__ == "__main__":
    launch()
