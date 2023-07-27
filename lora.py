import os, time, torch
import gradio as gr
from subprocess import getoutput
from diffusers import StableDiffusionPipeline
from gradio import strings
from trainer import Trainer

class Lora(Trainer):
    def tab():
        with gr.Tab("Train LoRA for WebUI"):
            with gr.Tab("Tag Images"):
                with gr.Box():
                    with gr.Accordion("Train Lora WebUI Tag Images Common Arguments", open=False):
                        gr.Markdown(
                        """
                        ```py
                        /content/drive/MyDrive/AI/training/parkminyoung \\
                        --repo_id SmilingWolf/wd-v1-4-convnext-tagger-v2 \\
                        --model_dir wd14_tagger_model \\
                        --thresh 0.35 \\
                        --batch_size 1 \\
                        --caption_extension .txt \\
                        ```
                        """)                        
                    train_tag_lora_webui_command = """python -u /content/trainer/sd-scripts/finetune/tag_images_by_wd14_tagger.py \\
                    /content/drive/MyDrive/AI/training/parkminyoung \\
                    --repo_id SmilingWolf/wd-v1-4-convnext-tagger-v2 \\
                    --model_dir wd14_tagger_model \\
                    --thresh 0.35 \\
                    --batch_size 1 \\
                    --caption_extension .txt"""
                    tag_lora_webui_command = gr.Textbox(show_label=False, lines=16, value=train_tag_lora_webui_command)
                    train_tag_lora_webui_out_text = gr.Textbox(show_label=False)
                    btn_train_tag_lora_webui_run_live = gr.Button("Train Lora")
                    btn_train_tag_lora_webui_run_live.click(Lora.run_live, inputs=tag_lora_webui_command, outputs=train_tag_lora_webui_out_text, show_progress=False)
            with gr.Tab("Merge Tags"):
                with gr.Box():
                    with gr.Accordion("Train Lora WebUI Merge Tags Common Arguments", open=False):
                        gr.Markdown(
                        """
                        ```py
                        /content/drive/MyDrive/AI/training/parkminyoung \\
                        /content/drive/MyDrive/AI/training/parkminyoung/parkminyoung.json \\
                        --caption_extension .txt \\
                        ```
                        """)                        
                    train_merge_lora_webui_command = """python -u /content/trainer/sd-scripts/finetune/merge_dd_tags_to_metadata.py \\
                    /content/drive/MyDrive/AI/training/parkminyoung \\
                    /content/drive/MyDrive/AI/training/parkminyoung/parkminyoung.json \\
                    --caption_extension .txt"""
                    merge_lora_webui_command = gr.Textbox(show_label=False, lines=16, value=train_merge_lora_webui_command)
                    train_merge_lora_webui_out_text = gr.Textbox(show_label=False)
                    btn_train_merge_lora_webui_run_live = gr.Button("Train Lora")
                    btn_train_merge_lora_webui_run_live.click(Lora.run_live, inputs=merge_lora_webui_command, outputs=train_merge_lora_webui_out_text, show_progress=False)
            with gr.Tab("Prepare Latents"):
                with gr.Box():
                    with gr.Accordion("Train Lora WebUI Merge Tags Common Arguments", open=False):
                        gr.Markdown(
                        """
                        ```py
                        /content/drive/MyDrive/AI/training/parkminyoung \\
                        /content/drive/MyDrive/AI/training/parkminyoung/parkminyoung.json \\
                        /content/drive/MyDrive/AI/training/parkminyoung/parkminyoung-latents.json \\
                        /content/model/model.ckpt \\
                        --batch_size 1 \\
                        --max_resolution 512,512 \\
                        --min_bucket_reso 256 \\
                        --max_bucket_reso 1024 \\
                        --bucket_reso_steps 64 \\
                        --mixed_precision no \\
                        ```
                        """)                        
                    train_prepare_lora_webui_command = """python -u /content/trainer/sd-scripts/finetune/prepare_buckets_latents.py \\
                    /content/drive/MyDrive/AI/training/parkminyoung \\
                    /content/drive/MyDrive/AI/training/parkminyoung/parkminyoung.json \\
                    /content/drive/MyDrive/AI/training/parkminyoung/parkminyoung-latents.json \\
                    /content/model/model.ckpt \\
                    --batch_size 1 \\
                    --max_resolution 512,512 \\
                    --min_bucket_reso 256 \\
                    --max_bucket_reso 1024 \\
                    --bucket_reso_steps 64 \\
                    --mixed_precision no"""
                    prepare_lora_webui_command = gr.Textbox(show_label=False, lines=16, value=train_prepare_lora_webui_command)
                    train_prepare_lora_webui_out_text = gr.Textbox(show_label=False)
                    btn_train_prepare_lora_webui_run_live = gr.Button("Train Lora")
                    btn_train_prepare_lora_webui_run_live.click(Lora.run_live, inputs=prepare_lora_webui_command, outputs=train_prepare_lora_webui_out_text, show_progress=False)
            with gr.Tab("Train"):
                    with gr.Box():
                        with gr.Accordion("Train Lora WebUI Common Arguments", open=False):
                            gr.Markdown(
                            """
                            ```py
                            --pretrained_model_name_or_path /content/model/model.ckpt \\
                            --train_data_dir /content/drive/MyDrive/AI/training/parkminyoung \\
                            --in_json /content/drive/MyDrive/AI/training/parkminyoung/parkminyoung-latents.json \\
                            --output_dir /content/trained \\
                            --xformers \\
                            --max_train_steps 1600 \\
                            --use_8bit_adam \\
                            --network_module networks.lora \\
                            ```
                            """)                        
                        train_lora_webui_command = """python -u /content/trainer/sd-scripts/train_network.py \\
                        --pretrained_model_name_or_path /content/model/model.ckpt \\
                        --train_data_dir /content/drive/MyDrive/AI/training/parkminyoung \\
                        --in_json /content/drive/MyDrive/AI/training/parkminyoung/parkminyoung-latents.json \\
                        --output_dir /content/trained \\
                        --xformers \\
                        --max_train_steps 1600 \\
                        --use_8bit_adam \\
                        --network_module networks.lora"""
                        lora_webui_command = gr.Textbox(show_label=False, lines=16, value=train_lora_webui_command)
                        train_lora_webui_out_text = gr.Textbox(show_label=False)
                        btn_train_lora_webui_run_live = gr.Button("Train Lora")
                        btn_train_lora_webui_run_live.click(Lora.run_live, inputs=lora_webui_command, outputs=train_lora_webui_out_text, show_progress=False)
            with gr.Tab("Tools"):
                with gr.Group():
                    with gr.Box():
                        with gr.Accordion("Remove Tags and Output Directory", open=False):
                            gr.Markdown(
                            """
                            ```py
                            rm /content/drive/MyDrive/AI/training/parkminyoung/*.txt && \\
                            rm /content/drive/MyDrive/AI/training/parkminyoung/*.npz && \\
                            rm /content/drive/MyDrive/AI/training/parkminyoung/parkminyoung.json && \\
                            rm /content/drive/MyDrive/AI/training/parkminyoung/parkminyoung-latents.json && \\
                            rm -rf /content/trained 
                            ```
                            """)
                        rm_lora_command = """rm /content/drive/MyDrive/AI/training/parkminyoung/*.txt && \\
                        rm /content/drive/MyDrive/AI/training/parkminyoung/*.npz && \\
                        rm /content/drive/MyDrive/AI/training/parkminyoung/parkminyoung.json && \\
                        rm /content/drive/MyDrive/AI/training/parkminyoung/parkminyoung-latents.json && \\
                        rm -rf /content/trained
                        """
                        rm_lora = gr.Textbox(show_label=False, lines=16, value=rm_lora_command)
                        rm_lora_out_text = gr.Textbox(show_label=False)
                        btn_run_static = gr.Button("Remove Lora Output Directory")
                        btn_run_static.click(Lora.run_live, inputs=rm_lora, outputs=rm_lora_out_text, show_progress=False)