import os, shutil
import gradio as gr
from gradio import strings
from shared import Shared

trainer = gr.Blocks(title="Trainer")

def upload_file(files):
    os.system(f"rm -rf /content/images")
    os.system(f"rm -rf /content/lora")
    os.system(f"rm -rf /content/dreambooth")
    file_paths = [file.name for file in files]
    if not os.path.exists('/content/images'):
        os.mkdir('/content/images')
        for file_path in file_paths:
            shutil.copy(file_path, '/content/images/')
    return file_paths

train_lora_command = f"""python -u /content/trainer/diffusers/lora/train_dreambooth_lora.py \\
                            --pretrained_model_name_or_path="/content/model"  \\
                            --instance_data_dir="/content/images" \\
                            --output_dir="/content/lora" \\
                            --learning_rate=5e-6 \\
                            --max_train_steps=1250 \\
                            --instance_prompt="⚠ Required" \\
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

def update_lora_instance_prompt(learning_rate, max_train_steps, instance_prompt):
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

train_dreambooth_command = """python -u /content/trainer/diffusers/dreambooth/train_dreambooth.py \\
                            --pretrained_model_name_or_path="/content/model"  \\
                            --instance_data_dir="/content/images" \\
                            --output_dir="/content/dreambooth" \\
                            --learning_rate=5e-6 \\
                            --max_train_steps=1250 \\
                            --instance_prompt="⚠ Required" \\
                            --resolution=512 \\
                            --center_crop \\
                            --train_batch_size=1 \\
                            --gradient_accumulation_steps=1 \\
                            --max_grad_norm=1.0 \\
                            --mixed_precision="fp16" \\
                            --gradient_checkpointing \\
                            --enable_xformers_memory_efficient_attention \\
                            --use_8bit_adam \\
                            --with_prior_preservation \\
                            --class_data_dir="/content/class_data_dir" \\
                            --prior_loss_weight=1.0 \\
                            --sample_batch_size=2 \\
                            --class_prompt="person" \\
                            --seed=69 \\
                            --num_class_images=12 \\
                            --offset_noise"""

def update_dreambooth_instance_prompt(learning_rate, max_train_steps, instance_prompt):
    train_dreambooth_command = f"""python -u /content/trainer/diffusers/dreambooth/train_dreambooth.py \\
                            --pretrained_model_name_or_path="/content/model"  \\
                            --instance_data_dir="/content/images" \\
                            --output_dir="/content/dreambooth" \\
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
                            --with_prior_preservation \\
                            --class_data_dir="/content/class_data_dir" \\
                            --prior_loss_weight=1.0 \\
                            --sample_batch_size=2 \\
                            --class_prompt="person" \\
                            --seed=69 \\
                            --num_class_images=12 \\
                            --offset_noise"""
    return train_dreambooth_command

def set_checkbox():
    return gr.Checkbox.update(value=False)

def set_textbox():
    return gr.Textbox.update(value="Training Done! 🥳")

def launch():
    strings.en["SHARE_LINK_MESSAGE"] = ""
    strings.en["BETA_INVITE"] = ""
    with trainer:
        with gr.Tab("Train Lora"):
            with gr.Row(equal_height=False):
                files_lora = gr.Files(label="Upload Images", file_types=["image"], file_count="multiple")
                files_lora.upload(fn=upload_file, inputs=files_lora)
                with gr.Box():
                    with gr.Group():
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
                        with gr.Row():
                            learning_rate_lora = gr.Textbox(label="Learning Rate", value=5e-6)
                            max_train_steps_lora = gr.Textbox(label="Max Train steps", value=1250)
                            instance_prompt_lora = gr.Textbox(label="Instance Prompt *", value="⚠ Required")
                        lora_command = gr.Textbox(show_label=False, lines=16, value=train_lora_command)
                        train_lora_out_text = gr.Textbox(show_label=False)
                        with gr.Row():
                            update_command_lora = gr.Button(value="Update train command")
                            btn_train_lora_run_live = gr.Button("Train Lora")
                            update_command_lora.click(fn=update_lora_instance_prompt, inputs=[learning_rate_lora, max_train_steps_lora, instance_prompt_lora], outputs=lora_command)
                            btn_train_lora_run_live.click(Shared.run_live, inputs=lora_command, outputs=train_lora_out_text, show_progress=True).then(set_textbox, None, train_lora_out_text, show_progress=False)
        with gr.Tab("Test Lora"):
            with gr.Row(equal_height=False):
                image_lora = gr.Image(show_label=False)
                with gr.Box():
                    with gr.Group():
                        model_dir_lora = gr.Textbox(label="Enter your output dir", show_label=False, max_lines=1, value="/content/model")
                        output_dir_lora = gr.Textbox(label="Enter your output dir", show_label=False, max_lines=1, value="/content/lora")
                        prompt_lora = gr.Textbox(label="prompt", show_label=False, max_lines=1, placeholder="Enter your prompt")
                        negative_prompt_lora = gr.Textbox(label="negative prompt", show_label=False, max_lines=1, placeholder="Enter your negative prompt")
                        steps_lora = gr.Slider(label="Steps", minimum=5, maximum=50, value=40, step=1)
                        scale_lora = gr.Slider(label="Guidance Scale", minimum=0, maximum=25, value=6, step=0.1)
                        checkbox_lora = gr.Checkbox(label="Load Model", value=True)
                        btn_test_lora = gr.Button("Generate image")
                        btn_test_lora.click(Shared.test_lora, inputs=[model_dir_lora, checkbox_lora, output_dir_lora, prompt_lora, negative_prompt_lora, steps_lora, scale_lora], outputs=image_lora).then(set_checkbox, None, checkbox_lora, show_progress=False)
        with gr.Tab("Train Dreambooth"):
            with gr.Row(equal_height=False):
                files_dreambooth = gr.Files(label="Upload Images", file_types=["image"], file_count="multiple")
                files_dreambooth.upload(fn=upload_file, inputs=files_dreambooth)
                with gr.Box():
                    with gr.Group():
                        with gr.Accordion("Train Dreambooth All Arguments", open=False):
                            gr.Markdown(
                            """
                            ```py
                            -h, --help            show this help message and exit
                            --pretrained_model_name_or_path PRETRAINED_MODEL_NAME_OR_PATH
                                                Path to pretrained model or model identifier from
                                                huggingface.co/models.
                            --revision REVISION   Revision of pretrained model identifier from
                                                huggingface.co/models. Trainable model components
                                                should be float32 precision.
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
                                                updates. Checkpoints can be used for resuming training
                                                via `--resume_from_checkpoint`. In the case that the
                                                checkpoint is better than the final trained model, the
                                                checkpoint can also be used for inference.Using a
                                                checkpoint for inference requires separate loading of
                                                the original pipeline and the individual checkpointed
                                                model components.See https://huggingface.co/docs/diffu
                                                sers/main/en/training/dreambooth#performing-inference-
                                                using-a-saved-checkpoint for step by stepinstructions.
                            --checkpoints_total_limit CHECKPOINTS_TOTAL_LIMIT
                                                Max number of checkpoints to store. Passed as
                                                `total_limit` to the `Accelerator`
                                                `ProjectConfiguration`. See Accelerator::save_state ht
                                                tps://huggingface.co/docs/accelerate/package_reference
                                                /accelerator#accelerate.Accelerator.save_state for
                                                more details
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
                            --use_8bit_adam       Whether or not to use 8-bit Adam from bitsandbytes.
                            --dataloader_num_workers DATALOADER_NUM_WORKERS
                                                Number of subprocesses to use for data loading. 0
                                                means that the data will be loaded in the main
                                                process.
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
                            --validation_prompt VALIDATION_PROMPT
                                                A prompt that is used during validation to verify that
                                                the model is learning.
                            --num_validation_images NUM_VALIDATION_IMAGES
                                                Number of images that should be generated during
                                                validation with `validation_prompt`.
                            --validation_steps VALIDATION_STEPS
                                                Run validation every X steps. Validation consists of
                                                running the prompt `args.validation_prompt` multiple
                                                times: `args.num_validation_images` and logging the
                                                images.
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
                            --set_grads_to_none   Save more memory by using setting grads to None
                                                instead of zero. Be aware, that this changes certain
                                                behaviors, so disable this argument if it causes any
                                                problems. More info: https://pytorch.org/docs/stable/g
                                                enerated/torch.optim.Optimizer.zero_grad.html
                            --offset_noise        Fine-tuning against a modified noise See:
                                                https://www.crosslabs.org//blog/diffusion-with-offset-
                                                noise for more information.
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
                            --skip_save_text_encoder
                                                Set to not save text encoder
                            --validation_images VALIDATION_IMAGES [VALIDATION_IMAGES ...]
                                                Optional set of images to use for validation. Used
                                                when the target pipeline takes an initial image as
                                                input such as when training image variation or
                                                superresolution.
                            --class_labels_conditioning CLASS_LABELS_CONDITIONING
                                                The optional `class_label` conditioning to pass to the
                                                unet, available values are `timesteps`.
                            ```
                            """)
                        with gr.Row():
                            learning_rate_dreambooth = gr.Textbox(label="Learning Rate", value=5e-6)
                            max_train_steps_dreambooth = gr.Textbox(label="Max Train steps", value=1250)
                            instance_prompt_dreambooth = gr.Textbox(label="Instance Prompt *", value="⚠ Required")
                        dreambooth_command = gr.Textbox(show_label=False, lines=24, value=train_dreambooth_command)
                        train_dreambooth_out_text = gr.Textbox(show_label=False)
                        with gr.Row():
                            update_command_dreambooth = gr.Button(value="Update train command")
                            btn_train_dreambooth_run_live = gr.Button("Train Dreambooth")
                            update_command_dreambooth.click(fn=update_dreambooth_instance_prompt, inputs=[learning_rate_dreambooth, max_train_steps_dreambooth, instance_prompt_dreambooth], outputs=dreambooth_command)
                            btn_train_dreambooth_run_live.click(Shared.run_live, inputs=dreambooth_command, outputs=train_dreambooth_out_text, show_progress=True).then(set_textbox, None, train_dreambooth_out_text, show_progress=False)
        with gr.Tab("Test Dreambooth"):
            with gr.Row(equal_height=False):
                image_dreambooth = gr.Image(show_label=False)
                with gr.Box():
                    with gr.Group():
                        output_dir_dreambooth = gr.Textbox(label="Enter your output dir", show_label=False, max_lines=1, value="/content/dreambooth")
                        prompt_dreambooth = gr.Textbox(label="prompt", show_label=False, max_lines=1, placeholder="Enter your prompt")
                        negative_prompt_dreambooth = gr.Textbox(label="negative prompt", show_label=False, max_lines=1, placeholder="Enter your negative prompt")
                        steps_dreambooth = gr.Slider(label="Steps", minimum=5, maximum=50, value=40, step=1)
                        scale_dreambooth = gr.Slider(label="Guidance Scale", minimum=0, maximum=25, value=6, step=0.1)
                        checkbox_dreambooth = gr.Checkbox(label="Load Model", value=True)
                        btn_test_dreambooth = gr.Button("Generate image")
                        btn_test_dreambooth.click(Shared.test_text_to_image, inputs=[output_dir_dreambooth, checkbox_dreambooth, prompt_dreambooth, negative_prompt_dreambooth, steps_dreambooth, scale_dreambooth], outputs=image_dreambooth).then(set_checkbox, None, checkbox_dreambooth, show_progress=False)
    trainer.queue().launch(debug=True, share=True, inline=False)

if __name__ == "__main__":
    launch()