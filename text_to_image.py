import gradio as gr
from shared import Shared

class TextToImage():
    def tab():
        with gr.Tab("Train Text to Image WebUI and Diffusers Lib"):
            with gr.Tab("Train"):
                with gr.Box():
                    with gr.Accordion("Train Text to Image Common Arguments", open=False):
                        gr.Markdown(
                        """
                        ```py
                        --pretrained_model_name_or_path="/content/model"  \\
                        --dataset_name="camenduru/test" \\
                        --use_ema \\
                        --train_data_dir="/content/images" \\
                        --output_dir="/content/trainer/diffusers/text_to_image/output_dir" \\
                        --learning_rate=1e-6 \\
                        --scale_lr \\
                        --lr_scheduler="constant" \\
                        --lr_warmup_steps=0 \\
                        --max_train_steps=5288 \\
                        --resolution=512 \\
                        --center_crop \\
                        --random_flip \\
                        --train_batch_size=1 \\
                        --gradient_accumulation_steps=1 \\
                        --max_grad_norm=1 \\
                        --mixed_precision="fp16" \\
                        --gradient_checkpointing \\
                        --enable_xformers_memory_efficient_attention \\
                        --use_8bit_adam \\
                        ```
                        """)
                    with gr.Accordion("Train Text to Image Common Arguments", open=False):
                        gr.Markdown(
                        """
                        ```py
                        -h, --help            show this help message and exit
                        --input_perturbation INPUT_PERTURBATION
                                              The scale of input perturbation. Recommended 0.1.
                        --pretrained_model_name_or_path PRETRAINED_MODEL_NAME_OR_PATH
                                              Path to pretrained model or model identifier from
                                              huggingface.co/models.
                        --revision REVISION   Revision of pretrained model identifier from
                                              huggingface.co/models.
                        --dataset_name DATASET_NAME
                                              The name of the Dataset (from the HuggingFace hub) to
                                              train on (could be your own, possibly private,
                                              dataset). It can also be a path pointing to a local
                                              copy of a dataset in your filesystem, or to a folder
                                              containing files that 🤗 Datasets can understand.
                        --dataset_config_name DATASET_CONFIG_NAME
                                              The config of the Dataset, leave as None if there's
                                              only one config.
                        --train_data_dir TRAIN_DATA_DIR
                                              A folder containing the training data. Folder contents
                                              must follow the structure described in https://hugging
                                              face.co/docs/datasets/image_dataset#imagefolder. In
                                              particular, a `metadata.jsonl` file must exist to
                                              provide the captions for the images. Ignored if
                                              `dataset_name` is specified.
                        --image_column IMAGE_COLUMN
                                              The column of the dataset containing an image.
                        --caption_column CAPTION_COLUMN
                                              The column of the dataset containing a caption or a
                                              list of captions.
                        --max_train_samples MAX_TRAIN_SAMPLES
                                              For debugging purposes or quicker training, truncate
                                              the number of training examples to this value if set.
                        --validation_prompts VALIDATION_PROMPTS [VALIDATION_PROMPTS ...]
                                              A set of prompts evaluated every `--validation_epochs`
                                              and logged to `--report_to`.
                        --output_dir OUTPUT_DIR
                                              The output directory where the model predictions and
                                              checkpoints will be written.
                        --cache_dir CACHE_DIR
                                              The directory where the downloaded models and datasets
                                              will be stored.
                        --seed SEED           A seed for reproducible training.
                        --resolution RESOLUTION
                                              The resolution for input images, all the images in the
                                              train/validation dataset will be resized to this
                                              resolution
                        --center_crop         Whether to center crop the input images to the
                                              resolution. If not set, the images will be randomly
                                              cropped. The images will be resized to the resolution
                                              first before cropping.
                        --random_flip         whether to randomly flip images horizontally
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
                        --snr_gamma SNR_GAMMA
                                              SNR weighting gamma to be used if rebalancing the
                                              loss. Recommended value is 5.0. More details here:
                                              https://arxiv.org/abs/2303.09556.
                        --use_8bit_adam       Whether or not to use 8-bit Adam from bitsandbytes.
                        --allow_tf32          Whether or not to allow TF32 on Ampere GPUs. Can be
                                              used to speed up training. For more information, see h
                                              ttps://pytorch.org/docs/stable/notes/cuda.html#tensorf
                                              loat-32-tf32-on-ampere-devices
                        --use_ema             Whether to use EMA model.
                        --non_ema_revision NON_EMA_REVISION
                                              Revision of pretrained non-ema model identifier. Must
                                              be a branch, tag or git identifier of the local or
                                              remote repository specified with
                                              --pretrained_model_name_or_path.
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
                        --prediction_type PREDICTION_TYPE
                                              The prediction_type that shall be used for training.
                                              Choose between 'epsilon' or 'v_prediction' or leave
                                              `None`. If left to `None` the default prediction type
                                              of the scheduler:
                                              `noise_scheduler.config.prediciton_type` is chosen.
                        --hub_model_id HUB_MODEL_ID
                                              The name of the repository to keep in sync with the
                                              local `output_dir`.
                        --logging_dir LOGGING_DIR
                                              [TensorBoard](https://www.tensorflow.org/tensorboard)
                                              log directory. Will default to
                                              *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***.
                        --mixed_precision {no,fp16,bf16}
                                              Whether to use mixed precision. Choose between fp16
                                              and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10.and
                                              an Nvidia Ampere GPU. Default to the value of
                                              accelerate config of the current system or the flag
                                              passed with the `accelerate.launch` command. Use this
                                              argument to override the accelerate config.
                        --report_to REPORT_TO
                                              The integration to report the results and logs to.
                                              Supported platforms are `"tensorboard"` (default),
                                              `"wandb"` and `"comet_ml"`. Use `"all"` to report to
                                              all integrations.
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
                        --noise_offset NOISE_OFFSET
                                              The scale of noise offset.
                        --validation_epochs VALIDATION_EPOCHS
                                              Run validation every X epochs.
                        --tracker_project_name TRACKER_PROJECT_NAME
                                              The `project_name` argument passed to
                                              Accelerator.init_trackers for more information see htt
                                              ps://huggingface.co/docs/accelerate/v0.17.0/en/package
                                              _reference/accelerator#accelerate.Accelerator
                        ```
                        """)
                    train_text_to_image_command = """python -u /content/trainer/diffusers/text_to_image/train_text_to_image.py \\
                    --pretrained_model_name_or_path="/content/model"  \\
                    --dataset_name="camenduru/test" \\
                    --use_ema \\
                    --train_data_dir="/content/images" \\
                    --output_dir="/content/trainer/diffusers/text_to_image/output_dir" \\
                    --learning_rate=1e-6 \\
                    --scale_lr \\
                    --lr_scheduler="constant" \\
                    --lr_warmup_steps=0 \\
                    --max_train_steps=5288 \\
                    --resolution=512 \\
                    --center_crop \\
                    --random_flip \\
                    --train_batch_size=1 \\
                    --gradient_accumulation_steps=1 \\
                    --max_grad_norm=1 \\
                    --mixed_precision="fp16" \\
                    --gradient_checkpointing \\
                    --enable_xformers_memory_efficient_attention \\
                    --use_8bit_adam"""
                    text_to_image_command = gr.Textbox(show_label=False, lines=23, value=train_text_to_image_command)
                    train_text_to_image_out_text = gr.Textbox(show_label=False)
                    btn_train_text_to_image_run_live = gr.Button("Train Textual Inversion")
                    btn_train_text_to_image_run_live.click(Shared.run_live, inputs=text_to_image_command, outputs=train_text_to_image_out_text, show_progress=False)
            with gr.Tab("Test"):
                with gr.Group():
                    with gr.Row():
                        with gr.Box():
                            image = gr.Image(show_label=False)
                        with gr.Box():
                            output_dir = gr.Textbox(label="Enter your output dir", show_label=False, max_lines=1, value="/content/trainer/diffusers/text_to_image/output_dir")
                            prompt = gr.Textbox(label="prompt", show_label=False, max_lines=1, placeholder="Enter your prompt")
                            negative_prompt = gr.Textbox(label="negative prompt", show_label=False, max_lines=1, placeholder="Enter your negative prompt")
                            steps = gr.Slider(label="Steps", minimum=5, maximum=50, value=25, step=1)
                            scale = gr.Slider(label="Guidance Scale", minimum=0, maximum=50, value=7.5, step=0.1)
                            checkbox = gr.Checkbox(label="Load Model", value=True)
                            btn_test_text_to_image = gr.Button("Generate image")
                            btn_test_text_to_image.click(Shared.test_text_to_image, inputs=[output_dir, checkbox, prompt, negative_prompt, steps, scale], outputs=image)
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
                        rm_text_to_image_command = """rm -rf /content/trainer/diffusers/textual_inversion/output_dir/*"""
                        rm_text_to_image = gr.Textbox(show_label=False, lines=1, value=rm_text_to_image_command)
                        rm_text_to_image_out_text = gr.Textbox(show_label=False)
                        btn_run_static = gr.Button("Remove Textual Inversion Output Directory")
                        btn_run_static.click(Shared.run_live, inputs=rm_text_to_image, outputs=rm_text_to_image_out_text, show_progress=False)