import os, shutil
import gradio as gr
from gradio import strings
from shared import Shared

trainer = gr.Blocks(title="Trainer")

train_lora_command = f"""python -u /content/trainer/diffusers/lora/train_dreambooth_lora.py \\
                            --pretrained_model_name_or_path="/content/model"  \\
                            --instance_data_dir="/content/images" \\
                            --output_dir="/content/lora" \\
                            --learning_rate=5e-6 \\
                            --max_train_steps=1250 \\
                            --instance_prompt="Required" \\
                            --resolution=512 \\
                            --center_crop \\
                            --train_batch_size=1 \\
                            --gradient_accumulation_steps=1 \\
                            --max_grad_norm=1.0 \\
                            --mixed_precision="fp16" \\
                            --gradient_checkpointing \\
                            --enable_xformers_memory_efficient_attention \\
                            --use_8bit_adam \\
                            --train_text_encoder"""

def upload_file(files):
    file_paths = [file.name for file in files]
    if not os.path.exists('/content/images'):
        os.rmdir("/content/images")
        os.mkdir('/content/images')
    for file_path in file_paths:
        shutil.copy(file_path, '/content/images/')
    return file_paths

def update_instance_prompt(learning_rate, max_train_steps, instance_prompt):
    train_lora_command = f"""python -u /content/trainer/diffusers/lora/train_dreambooth_lora.py \\
                            --pretrained_model_name_or_path="/content/model"  \\
                            --instance_data_dir="/content/images" \\
                            --output_dir="/content/lora" \\
                            --learning_rate={learning_rate} \\
                            --max_train_steps={max_train_steps} \\
                            --instance_prompt="{instance_prompt}" \\
                            --resolution=512 \\
                            --center_crop \\
                            --train_batch_size=1 \\
                            --gradient_accumulation_steps=1 \\
                            --max_grad_norm=1.0 \\
                            --mixed_precision="fp16" \\
                            --gradient_checkpointing \\
                            --enable_xformers_memory_efficient_attention \\
                            --use_8bit_adam \\
                            --train_text_encoder"""
    return train_lora_command

def launch():
    strings.en["SHARE_LINK_MESSAGE"] = ""
    strings.en["BETA_INVITE"] = ""
    with trainer:
        with gr.Tab("Train"):
          with gr.Row():
              with gr.Box():
                files = gr.Files(label="Upload Images", file_types=["image"], file_count="multiple")
                files.upload(fn=upload_file, inputs=files)
              with gr.Box():
                  learning_rate = gr.Textbox(label="Learning Rate", value=5e-6)
                  max_train_steps = gr.Textbox(label="Max Train steps", value=1250)
                  instance_prompt = gr.Textbox(label="Instance Prompt *", value="Required")
                  lora_command = gr.Textbox(show_label=False, lines=16, value=train_lora_command)
                  train_lora_out_text = gr.Textbox(show_label=False)
                  update_command = gr.Button(value="Update train command")
                  btn_train_lora_run_live = gr.Button("Train Lora")
                  update_command.click(fn=update_instance_prompt, inputs=[learning_rate, max_train_steps, instance_prompt], outputs=lora_command)
                  btn_train_lora_run_live.click(Shared.run_live, inputs=lora_command, outputs=train_lora_out_text, show_progress=False)
        with gr.Tab("Test"):
          with gr.Row():
              with gr.Box():
                  image = gr.Image(show_label=False)
              with gr.Box():
                  model_dir = gr.Textbox(label="Enter your output dir", show_label=False, max_lines=1, value="/content/model")
                  output_dir = gr.Textbox(label="Enter your output dir", show_label=False, max_lines=1, value="/content/lora")
                  prompt = gr.Textbox(label="prompt", show_label=False, max_lines=1, placeholder="Enter your prompt")
                  negative_prompt = gr.Textbox(label="negative prompt", show_label=False, max_lines=1, placeholder="Enter your negative prompt")
                  steps = gr.Slider(label="Steps", minimum=5, maximum=50, value=25, step=1)
                  scale = gr.Slider(label="Guidance Scale", minimum=0, maximum=50, value=7.5, step=0.1)
                  checkbox = gr.Checkbox(label="Load Model", value=True)
                  btn_test_lora = gr.Button("Generate image")
                  btn_test_lora.click(Shared.test_lora, inputs=[model_dir, checkbox, output_dir, prompt, negative_prompt, steps, scale], outputs=image)
    trainer.queue().launch(debug=True, share=True, inline=False)

if __name__ == "__main__":
    launch()