import gradio as gr
from shared import Shared

class LoraD():
    def tab():
        with gr.Tab("Train LoRA for Diffusers Lib"):
            with gr.Tab("Train"):
                with gr.Box():
                    with gr.Accordion("Train Lora Common Arguments", open=False):
                        gr.Markdown(
                        """
                        ```py
                        --pretrained_model_name_or_path="/content/model"  \\
                        --instance_data_dir="/content/images" \\
                        --output_dir="/content/trainer/diffusers/lora/output_dir" \\
                        --learning_rate=5e-6 \\
                        --max_train_steps=1250 \\
                        --instance_prompt="⚠ INSTANCE PROMPT" \\
                        --resolution=512 \\
                        --center_crop \\
                        --train_batch_size=1 \\
                        --gradient_accumulation_steps=1 \\
                        --max_grad_norm=1.0 \\
                        --mixed_precision="fp16" \\
                        --gradient_checkpointing \\
                        --enable_xformers_memory_efficient_attention \\
                        --use_8bit_adam \n
                        --with_prior_preservation \\
                        --class_data_dir="/content/trainer/diffusers/lora/class_data_dir" \\
                        --prior_loss_weight=1.0 \\
                        --sample_batch_size=2 \\
                        --class_prompt="person" \\
                        --seed=69 \\
                        --num_class_images=12 \\
                        --train_text_encoder \\
                        ```
                        """)
                    with gr.Accordion("Train Lora All Arguments", open=False):
                        gr.Markdown(
                        """
                        ```py
                        -h, --help            show this help message and exit
                        --pretrained_model_name_or_path PRETRAINED_MODEL_NAME_OR_PATH
                                            Path to pretrained model or model identifier from
                                            huggingface.co/models.
                        --revision REVISION   Revision of pretrained model identifier from
                                            huggingface.co/models.
                        --tokenizer_name TOKENIZER_NAME
                                            Pretrained tokenizer name or path if not the same as
                                            model_name
                        --instance_data_dir INSTANCE_DATA_DIR
                                            A folder containing the training data of instance
                                            images.
                        --class_data_dir CLASS_DATA_DIR
                                            A folder containing the training data of class images.
                        --instance_prompt INSTANCE_PROMPT
                                            The prompt with identifier specifying the instance
                        --class_prompt CLASS_PROMPT
                                            The prompt to specify images in the same class as
                                            provided instance images.
                        --validation_prompt VALIDATION_PROMPT
                                            A prompt that is used during validation to verify that
                                            the model is learning.
                        --num_validation_images NUM_VALIDATION_IMAGES
                                            Number of images that should be generated during
                                            validation with `validation_prompt`.
                        --validation_epochs VALIDATION_EPOCHS
                                            Run dreambooth validation every X epochs. Dreambooth
                                            validation consists of running the prompt
                                            `args.validation_prompt` multiple times:
                                            `args.num_validation_images`.
                        --with_prior_preservation
                                            Flag to add prior preservation loss.
                        --prior_loss_weight PRIOR_LOSS_WEIGHT
                                            The weight of prior preservation loss.
                        --num_class_images NUM_CLASS_IMAGES
                                            Minimal class images for prior preservation loss. If
                                            there are not enough images already present in
                                            class_data_dir, additional images will be sampled with
                                            class_prompt.
                        --output_dir OUTPUT_DIR
                                            The output directory where the model predictions and
                                            checkpoints will be written.
                        --seed SEED           A seed for reproducible training.
                        --resolution RESOLUTION
                                            The resolution for input images, all the images in the
                                            train/validation dataset will be resized to this
                                            resolution
                        --center_crop         Whether to center crop the input images to the
                                            resolution. If not set, the images will be randomly
                                            cropped. The images will be resized to the resolution
                                            first before cropping.
                        --train_text_encoder  Whether to train the text encoder. If set, the text
                                            encoder should be float32 precision.
                        --train_batch_size TRAIN_BATCH_SIZE
                                            Batch size (per device) for the training dataloader.
                        --sample_batch_size SAMPLE_BATCH_SIZE
                                            Batch size (per device) for sampling images.
                        --num_train_epochs NUM_TRAIN_EPOCHS
                        --max_train_steps MAX_TRAIN_STEPS
                                            Total number of training steps to perform. If
                                            provided, overrides num_train_epochs.
                        --checkpointing_steps CHECKPOINTING_STEPS
                                            Save a checkpoint of the training state every X
                                            updates. These checkpoints can be used both as final
                                            checkpoints in case they are better than the last
                                            checkpoint, and are also suitable for resuming
                                            training using `--resume_from_checkpoint`.
                        --checkpoints_total_limit CHECKPOINTS_TOTAL_LIMIT
                                            Max number of checkpoints to store.
                        --resume_from_checkpoint RESUME_FROM_CHECKPOINT
                                            Whether training should be resumed from a previous
                                            checkpoint. Use a path saved by
                                            `--checkpointing_steps`, or `"latest"` to
                                            automatically select the last available checkpoint.
                        --gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS
                                            Number of updates steps to accumulate before
                                            performing a backward/update pass.
                        --gradient_checkpointing
                                            Whether or not to use gradient checkpointing to save
                                            memory at the expense of slower backward pass.
                        --learning_rate LEARNING_RATE
                                            Initial learning rate (after the potential warmup
                                            period) to use.
                        --scale_lr            Scale the learning rate by the number of GPUs,
                                            gradient accumulation steps, and batch size.
                        --lr_scheduler LR_SCHEDULER
                                            The scheduler type to use. Choose between ["linear",
                                            "cosine", "cosine_with_restarts", "polynomial",
                                            "constant", "constant_with_warmup"]
                        --lr_warmup_steps LR_WARMUP_STEPS
                                            Number of steps for the warmup in the lr scheduler.
                        --lr_num_cycles LR_NUM_CYCLES
                                            Number of hard resets of the lr in
                                            cosine_with_restarts scheduler.
                        --lr_power LR_POWER   Power factor of the polynomial scheduler.
                        --dataloader_num_workers DATALOADER_NUM_WORKERS
                                            Number of subprocesses to use for data loading. 0
                                            means that the data will be loaded in the main
                                            process.
                        --use_8bit_adam       Whether or not to use 8-bit Adam from bitsandbytes.
                        --adam_beta1 ADAM_BETA1
                                            The beta1 parameter for the Adam optimizer.
                        --adam_beta2 ADAM_BETA2
                                            The beta2 parameter for the Adam optimizer.
                        --adam_weight_decay ADAM_WEIGHT_DECAY
                                            Weight decay to use.
                        --adam_epsilon ADAM_EPSILON
                                            Epsilon value for the Adam optimizer
                        --max_grad_norm MAX_GRAD_NORM
                                            Max gradient norm.
                        --push_to_hub         Whether or not to push the model to the Hub.
                        --hub_token HUB_TOKEN
                                            The token to use to push to the Model Hub.
                        --hub_model_id HUB_MODEL_ID
                                            The name of the repository to keep in sync with the
                                            local `output_dir`.
                        --logging_dir LOGGING_DIR
                                            [TensorBoard](https://www.tensorflow.org/tensorboard)
                                            log directory. Will default to
                                            *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***.
                        --allow_tf32          Whether or not to allow TF32 on Ampere GPUs. Can be
                                            used to speed up training. For more information, see h
                                            ttps://pytorch.org/docs/stable/notes/cuda.html#tensorf
                                            loat-32-tf32-on-ampere-devices
                        --report_to REPORT_TO
                                            The integration to report the results and logs to.
                                            Supported platforms are `"tensorboard"` (default),
                                            `"wandb"` and `"comet_ml"`. Use `"all"` to report to
                                            all integrations.
                        --mixed_precision {no,fp16,bf16}
                                            Whether to use mixed precision. Choose between fp16
                                            and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10.and
                                            an Nvidia Ampere GPU. Default to the value of
                                            accelerate config of the current system or the flag
                                            passed with the `accelerate.launch` command. Use this
                                            argument to override the accelerate config.
                        --prior_generation_precision {no,fp32,fp16,bf16}
                                            Choose prior generation precision between fp32, fp16
                                            and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10.and
                                            an Nvidia Ampere GPU. Default to fp16 if a GPU is
                                            available else fp32.
                        --local_rank LOCAL_RANK
                                            For distributed training: local_rank
                        --enable_xformers_memory_efficient_attention
                                            Whether or not to use xformers.
                        --pre_compute_text_embeddings
                                            Whether or not to pre-compute text embeddings. If text
                                            embeddings are pre-computed, the text encoder will not
                                            be kept in memory during training and will leave more
                                            GPU memory available for training the rest of the
                                            model. This is not compatible with
                                            `--train_text_encoder`.
                        --tokenizer_max_length TOKENIZER_MAX_LENGTH
                                            The maximum length of the tokenizer. If not set, will
                                            default to the tokenizer's max length.
                        --text_encoder_use_attention_mask
                                            Whether to use attention mask for the text encoder
                        --validation_images VALIDATION_IMAGES [VALIDATION_IMAGES ...]
                                            Optional set of images to use for validation. Used
                                            when the target pipeline takes an initial image as
                                            input such as when training image variation or
                                            superresolution.
                        --class_labels_conditioning CLASS_LABELS_CONDITIONING
                                            The optional `class_label` conditioning to pass to the
                                            unet, available values are `timesteps`.
                        --rank RANK         The dimension of the LoRA update matrices.
                        ```
                        """)
                    train_lora_command = """python -u /content/trainer/diffusers/lora/train_dreambooth_lora.py \\
                    --pretrained_model_name_or_path="/content/model"  \\
                    --instance_data_dir="/content/images" \\
                    --output_dir="/content/trainer/diffusers/lora/output_dir" \\
                    --learning_rate=5e-6 \\
                    --max_train_steps=1250 \\
                    --instance_prompt="⚠ INSTANCE PROMPT" \\
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
                    lora_command = gr.Textbox(show_label=False, lines=16, value=train_lora_command)
                    train_lora_out_text = gr.Textbox(show_label=False)
                    btn_train_lora_run_live = gr.Button("Train Lora")
                    btn_train_lora_run_live.click(Shared.run_live, inputs=lora_command, outputs=train_lora_out_text, show_progress=False)
            with gr.Tab("Test"):
                with gr.Group():
                    with gr.Row():
                        with gr.Box():
                            image = gr.Image(show_label=False)
                        with gr.Box():
                            model_dir = gr.Textbox(label="Enter your output dir", show_label=False, max_lines=1, value="/content/model")
                            output_dir = gr.Textbox(label="Enter your output dir", show_label=False, max_lines=1, value="/content/trainer/diffusers/lora/output_dir")
                            prompt = gr.Textbox(label="prompt", show_label=False, max_lines=1, placeholder="Enter your prompt")
                            negative_prompt = gr.Textbox(label="negative prompt", show_label=False, max_lines=1, placeholder="Enter your negative prompt")
                            steps = gr.Slider(label="Steps", minimum=5, maximum=50, value=25, step=1)
                            scale = gr.Slider(label="Guidance Scale", minimum=0, maximum=50, value=7.5, step=0.1)
                            checkbox = gr.Checkbox(label="Load Model", value=True)
                            btn_test_lora = gr.Button("Generate image")
                            btn_test_lora.click(Shared.test_lora, inputs=[model_dir, checkbox, output_dir, prompt, negative_prompt, steps, scale], outputs=image) 
            with gr.Tab("Tools"):
                with gr.Group():
                    with gr.Box():
                        with gr.Accordion("Copy Lora to Additional Network", open=False):
                            gr.Markdown(
                            """
                            ```py
                            cp /content/trainer/diffusers/lora/output_dir/pytorch_lora_weights.safetensors \\
                            /content/stable-diffusion-webui/extensions/sd-webui-additional-networks/models/lora/instance_prompt.safetensors
                            ```
                            """)
                        cp_lora_command = """cp /content/trainer/diffusers/lora/output_dir/pytorch_lora_weights.safetensors \\
                        /content/stable-diffusion-webui/extensions/sd-webui-additional-networks/models/lora/instance_prompt.safetensors"""
                        cp_lora = gr.Textbox(show_label=False, lines=2, value=cp_lora_command)
                        cp_lora_out_text = gr.Textbox(show_label=False)
                        btn_run_static = gr.Button("Copy Lora to Additional Network")
                        btn_run_static.click(Shared.run_live, inputs=cp_lora, outputs=cp_lora_out_text, show_progress=False)
                with gr.Group():
                    with gr.Box():
                        with gr.Accordion("Remove Lora Output Directory", open=False):
                            gr.Markdown(
                            """
                            ```py
                            rm -rf /content/trainer/diffusers/lora/output_dir/*
                            ```
                            """)
                        rm_lora_command = """rm -rf /content/trainer/diffusers/lora/output_dir/*"""
                        rm_lora = gr.Textbox(show_label=False, lines=1, value=rm_lora_command)
                        rm_lora_out_text = gr.Textbox(show_label=False)
                        btn_run_static = gr.Button("Remove Lora Output Directory")
                        btn_run_static.click(Shared.run_live, inputs=rm_lora, outputs=rm_lora_out_text, show_progress=False)