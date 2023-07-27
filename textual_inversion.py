import gradio as gr
from shared import Shared

class TextualInversion():
    def tab():
        with gr.Tab("Textual Inversion for WebUI and Diffusers Lib"):
            with gr.Tab("Train"):
                with gr.Box():
                    with gr.Accordion("Train Textual Inversion Common Arguments", open=False):
                        gr.Markdown(
                        """
                        ```py
                        --pretrained_model_name_or_path="/content/model"  \\
                        --instance_data_dir="/content/images" \\
                        --output_dir="/content/trainer/diffusers/dreambooth/output_dir" \\
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
                        --with_prior_preservation \\
                        --class_data_dir="/content/trainer/diffusers/dreambooth/class_data_dir" \\
                        --prior_loss_weight=1.0 \\
                        --sample_batch_size=2 \\
                        --class_prompt="person" \\
                        --seed=69 \\
                        --num_class_images=12 \\
                        ```
                        """)
                    with gr.Accordion("Train Textual Inversion All Arguments", open=False):
                        gr.Markdown(
                        """
                        ```py
                        -h, --help            show this help message and exit
                        --save_steps SAVE_STEPS
                                              Save learned_embeds.bin every X updates steps.
                        --save_as_full_pipeline
                                              Save the complete stable diffusion pipeline.
                        --num_vectors NUM_VECTORS
                                              How many textual inversion vectors shall be used to
                                              learn the concept.
                        --pretrained_model_name_or_path PRETRAINED_MODEL_NAME_OR_PATH
                                              Path to pretrained model or model identifier from
                                              huggingface.co/models.
                        --revision REVISION   Revision of pretrained model identifier from
                                              huggingface.co/models.
                        --tokenizer_name TOKENIZER_NAME
                                              Pretrained tokenizer name or path if not the same as
                                              model_name
                        --train_data_dir TRAIN_DATA_DIR
                                              A folder containing the training data.
                        --placeholder_token PLACEHOLDER_TOKEN
                                              A token to use as a placeholder for the concept.
                        --initializer_token INITIALIZER_TOKEN
                                              A token to use as initializer word.
                        --learnable_property LEARNABLE_PROPERTY
                                              Choose between 'object' and 'style'
                        --repeats REPEATS     How many times to repeat the training data.
                        --output_dir OUTPUT_DIR
                                              The output directory where the model predictions and
                                              checkpoints will be written.
                        --seed SEED           A seed for reproducible training.
                        --resolution RESOLUTION
                                              The resolution for input images, all the images in the
                                              train/validation dataset will be resized to this
                                              resolution
                        --center_crop         Whether to center crop images before resizing to
                                              resolution.
                        --train_batch_size TRAIN_BATCH_SIZE
                                              Batch size (per device) for the training dataloader.
                        --num_train_epochs NUM_TRAIN_EPOCHS
                        --max_train_steps MAX_TRAIN_STEPS
                                              Total number of training steps to perform. If
                                              provided, overrides num_train_epochs.
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
                        --mixed_precision {no,fp16,bf16}
                                              Whether to use mixed precision. Choosebetween fp16 and
                                              bf16 (bfloat16). Bf16 requires PyTorch >= 1.10.and an
                                              Nvidia Ampere GPU.
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
                        --validation_epochs VALIDATION_EPOCHS
                                              Deprecated in favor of validation_steps. Run
                                              validation every X epochs. Validation consists of
                                              running the prompt `args.validation_prompt` multiple
                                              times: `args.num_validation_images` and logging the
                                              images.
                        --local_rank LOCAL_RANK
                                              For distributed training: local_rank
                        --checkpointing_steps CHECKPOINTING_STEPS
                                              Save a checkpoint of the training state every X
                                              updates. These checkpoints are only suitable for
                                              resuming training using `--resume_from_checkpoint`.
                        --checkpoints_total_limit CHECKPOINTS_TOTAL_LIMIT
                                              Max number of checkpoints to store.
                        --resume_from_checkpoint RESUME_FROM_CHECKPOINT
                                              Whether training should be resumed from a previous
                                              checkpoint. Use a path saved by
                                              `--checkpointing_steps`, or `"latest"` to
                                              automatically select the last available checkpoint.
                        --enable_xformers_memory_efficient_attention
                                              Whether or not to use xformers.
                        ```
                        """)
                    train_textual_inversion_command = """python -u /content/trainer/diffusers/textual_inversion/textual_inversion.py \\
                    --pretrained_model_name_or_path="/content/model"  \\
                    --train_data_dir="/content/images" \\
                    --learnable_property="object" \\
                    --output_dir="/content/trainer/diffusers/textual_inversion/output_dir" \\
                    --placeholder_token="<instance_prompt>" \\
                    --initializer_token="⚠ INSTANCE PROMPT" \\
                    --learning_rate=5e-6 \\
                    --scale_lr \\
                    --lr_scheduler="constant" \\
                    --lr_warmup_steps=0 \\
                    --max_train_steps=3000 \\
                    --resolution=512 \\
                    --center_crop \\
                    --train_batch_size=1 \\
                    --gradient_accumulation_steps=1 \\
                    --mixed_precision="fp16" \\
                    --gradient_checkpointing \\
                    --enable_xformers_memory_efficient_attention"""
                    textual_inversion_command = gr.Textbox(show_label=False, lines=23, value=train_textual_inversion_command)
                    train_textual_inversion_out_text = gr.Textbox(show_label=False)
                    btn_train_textual_inversion_run_live = gr.Button("Train Textual Inversion")
                    btn_train_textual_inversion_run_live.click(Shared.run_live, inputs=textual_inversion_command, outputs=train_textual_inversion_out_text, show_progress=False)
            with gr.Tab("Test"):
                with gr.Group():
                    with gr.Row():
                        with gr.Box():
                            image = gr.Image(show_label=False)
                        with gr.Box():
                            output_dir = gr.Textbox(label="Enter your output dir", show_label=False, max_lines=1, value="/content/trainer/diffusers/dreambooth/output_dir")
                            prompt = gr.Textbox(label="prompt", show_label=False, max_lines=1, placeholder="Enter your prompt")
                            negative_prompt = gr.Textbox(label="negative prompt", show_label=False, max_lines=1, placeholder="Enter your negative prompt")
                            steps = gr.Slider(label="Steps", minimum=5, maximum=50, value=25, step=1)
                            scale = gr.Slider(label="Guidance Scale", minimum=0, maximum=50, value=7.5, step=0.1)
                            checkbox = gr.Checkbox(label="Load Model", value=True)
                            btn_test_dreambooth = gr.Button("Generate image")
                            btn_test_dreambooth.click(Shared.test_text_to_image, inputs=[output_dir, checkbox, prompt, negative_prompt, steps, scale], outputs=image)
            with gr.Tab("Tools"):
                with gr.Group():
                    with gr.Box():
                        with gr.Accordion("Remove Textual Inversion Output Directory", open=False):
                            gr.Markdown(
                            """
                            ```py
                            rm -rf /content/trainer/diffusers/textual_inversion/output_dir/*
                            ```
                            """)
                        rm_dreambooth_command = """rm -rf /content/trainer/diffusers/textual_inversion/output_dir/*"""
                        rm_dreambooth = gr.Textbox(show_label=False, lines=1, value=rm_dreambooth_command)
                        rm_dreambooth_out_text = gr.Textbox(show_label=False)
                        btn_run_static = gr.Button("Remove Textual Inversion Output Directory")
                        btn_run_static.click(Shared.run_live, inputs=rm_dreambooth, outputs=rm_dreambooth_out_text, show_progress=False)