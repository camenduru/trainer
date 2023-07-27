import os, time, torch
import gradio as gr
from subprocess import getoutput
from diffusers import StableDiffusionPipeline
from gradio import strings

from dreambooth import Dreambooth

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
                        --train_data_dir="/content/drive/MyDrive/AI/training/parkminyoung" \\
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
                                                Max number of checkpoints to store. Passed as
                                                `total_limit` to the `Accelerator`
                                                `ProjectConfiguration`. See Accelerator::save_state ht
                                                tps://huggingface.co/docs/accelerate/package_reference
                                                /accelerator#accelerate.Accelerator.save_state for
                                                more docs
                        --resume_from_checkpoint RESUME_FROM_CHECKPOINT
                                                Whether training should be resumed from a previous
                                                checkpoint. Use a path saved by
                                                `--checkpointing_steps`, or `"latest"` to
                                                automatically select the last available checkpoint.
                        --enable_xformers_memory_efficient_attention
                                                Whether or not to use xformers.
                        ```
                        """)
                    train_text_to_image_command = """python -u /content/trainer/diffusers/text_to_image/train_text_to_image.py \\
                    --pretrained_model_name_or_path="/content/model"  \\
                    --dataset_name="camenduru/test" \\
                    --use_ema \\
                    --train_data_dir="/content/drive/MyDrive/AI/training/parkminyoung" \\
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
                    btn_train_text_to_image_run_live.click(run_live, inputs=text_to_image_command, outputs=train_text_to_image_out_text, show_progress=False)
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
                            btn_test_text_to_image.click(test_text_to_image, inputs=[output_dir, checkbox, prompt, negative_prompt, steps, scale], outputs=image)
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
                        btn_run_static.click(run_live, inputs=rm_text_to_image, outputs=rm_text_to_image_out_text, show_progress=False)
        # with gr.Tab("Train LoRA for Diffusers Lib"):
        #     with gr.Tab("Train"):
        #         with gr.Box():
        #             with gr.Accordion("Train Lora Common Arguments", open=False):
        #                 gr.Markdown(
        #                 """
        #                 ```py
        #                 --pretrained_model_name_or_path="/content/model"  \\
        #                 --instance_data_dir="/content/drive/MyDrive/AI/training/parkminyoung" \\
        #                 --output_dir="/content/trainer/diffusers/lora/output_dir" \\
        #                 --learning_rate=5e-6 \\
        #                 --max_train_steps=650 \\
        #                 --instance_prompt="parkminyoung" \\
        #                 --resolution=512 \\
        #                 --center_crop \\
        #                 --train_batch_size=1 \\
        #                 --gradient_accumulation_steps=1 \\
        #                 --max_grad_norm=1.0 \\
        #                 --mixed_precision="fp16" \\
        #                 --gradient_checkpointing \\
        #                 --enable_xformers_memory_efficient_attention \\
        #                 --use_8bit_adam \n
        #                 --with_prior_preservation \\
        #                 --class_data_dir="/content/trainer/diffusers/lora/class_data_dir" \\
        #                 --prior_loss_weight=1.0 \\
        #                 --sample_batch_size=2 \\
        #                 --class_prompt="person" \\
        #                 --seed=69 \\
        #                 --num_class_images=12 \\
        #                 ```
        #                 """)
        #             with gr.Accordion("Train Lora All Arguments", open=False):
        #                 gr.Markdown(
        #                 """
        #                 ```py
        #                 -h, --help            show this help message and exit
        #                 --pretrained_model_name_or_path PRETRAINED_MODEL_NAME_OR_PATH
        #                                         Path to pretrained model or model identifier from
        #                                         huggingface.co/models.
        #                 --revision REVISION   Revision of pretrained model identifier from
        #                                         huggingface.co/models.
        #                 --tokenizer_name TOKENIZER_NAME
        #                                         Pretrained tokenizer name or path if not the same as
        #                                         model_name
        #                 --instance_data_dir INSTANCE_DATA_DIR
        #                                         A folder containing the training data of instance
        #                                         images.
        #                 --class_data_dir CLASS_DATA_DIR
        #                                         A folder containing the training data of class images.
        #                 --instance_prompt INSTANCE_PROMPT
        #                                         The prompt with identifier specifying the instance
        #                 --class_prompt CLASS_PROMPT
        #                                         The prompt to specify images in the same class as
        #                                         provided instance images.
        #                 --validation_prompt VALIDATION_PROMPT
        #                                         A prompt that is used during validation to verify that
        #                                         the model is learning.
        #                 --num_validation_images NUM_VALIDATION_IMAGES
        #                                         Number of images that should be generated during
        #                                         validation with `validation_prompt`.
        #                 --validation_epochs VALIDATION_EPOCHS
        #                                         Run dreambooth validation every X epochs. Dreambooth
        #                                         validation consists of running the prompt
        #                                         `args.validation_prompt` multiple times:
        #                                         `args.num_validation_images`.
        #                 --with_prior_preservation
        #                                         Flag to add prior preservation loss.
        #                 --prior_loss_weight PRIOR_LOSS_WEIGHT
        #                                         The weight of prior preservation loss.
        #                 --num_class_images NUM_CLASS_IMAGES
        #                                         Minimal class images for prior preservation loss. If
        #                                         there are not enough images already present in
        #                                         class_data_dir, additional images will be sampled with
        #                                         class_prompt.
        #                 --output_dir OUTPUT_DIR
        #                                         The output directory where the model predictions and
        #                                         checkpoints will be written.
        #                 --seed SEED           A seed for reproducible training.
        #                 --resolution RESOLUTION
        #                                         The resolution for input images, all the images in the
        #                                         train/validation dataset will be resized to this
        #                                         resolution
        #                 --center_crop         Whether to center crop the input images to the
        #                                         resolution. If not set, the images will be randomly
        #                                         cropped. The images will be resized to the resolution
        #                                         first before cropping.
        #                 --train_batch_size TRAIN_BATCH_SIZE
        #                                         Batch size (per device) for the training dataloader.
        #                 --sample_batch_size SAMPLE_BATCH_SIZE
        #                                         Batch size (per device) for sampling images.
        #                 --num_train_epochs NUM_TRAIN_EPOCHS
        #                 --max_train_steps MAX_TRAIN_STEPS
        #                                         Total number of training steps to perform. If
        #                                         provided, overrides num_train_epochs.
        #                 --checkpointing_steps CHECKPOINTING_STEPS
        #                                         Save a checkpoint of the training state every X
        #                                         updates. These checkpoints can be used both as final
        #                                         checkpoints in case they are better than the last
        #                                         checkpoint, and are also suitable for resuming
        #                                         training using `--resume_from_checkpoint`.
        #                 --checkpoints_total_limit CHECKPOINTS_TOTAL_LIMIT
        #                                         Max number of checkpoints to store. Passed as
        #                                         `total_limit` to the `Accelerator`
        #                                         `ProjectConfiguration`. See Accelerator::save_state ht
        #                                         tps://huggingface.co/docs/accelerate/package_reference
        #                                         /accelerator#accelerate.Accelerator.save_state for
        #                                         more docs
        #                 --resume_from_checkpoint RESUME_FROM_CHECKPOINT
        #                                         Whether training should be resumed from a previous
        #                                         checkpoint. Use a path saved by
        #                                         `--checkpointing_steps`, or `"latest"` to
        #                                         automatically select the last available checkpoint.
        #                 --gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS
        #                                         Number of updates steps to accumulate before
        #                                         performing a backward/update pass.
        #                 --gradient_checkpointing
        #                                         Whether or not to use gradient checkpointing to save
        #                                         memory at the expense of slower backward pass.
        #                 --learning_rate LEARNING_RATE
        #                                         Initial learning rate (after the potential warmup
        #                                         period) to use.
        #                 --scale_lr            Scale the learning rate by the number of GPUs,
        #                                         gradient accumulation steps, and batch size.
        #                 --lr_scheduler LR_SCHEDULER
        #                                         The scheduler type to use. Choose between ["linear",
        #                                         "cosine", "cosine_with_restarts", "polynomial",
        #                                         "constant", "constant_with_warmup"]
        #                 --lr_warmup_steps LR_WARMUP_STEPS
        #                                         Number of steps for the warmup in the lr scheduler.
        #                 --lr_num_cycles LR_NUM_CYCLES
        #                                         Number of hard resets of the lr in
        #                                         cosine_with_restarts scheduler.
        #                 --lr_power LR_POWER   Power factor of the polynomial scheduler.
        #                 --dataloader_num_workers DATALOADER_NUM_WORKERS
        #                                         Number of subprocesses to use for data loading. 0
        #                                         means that the data will be loaded in the main
        #                                         process.
        #                 --use_8bit_adam       Whether or not to use 8-bit Adam from bitsandbytes.
        #                 --adam_beta1 ADAM_BETA1
        #                                         The beta1 parameter for the Adam optimizer.
        #                 --adam_beta2 ADAM_BETA2
        #                                         The beta2 parameter for the Adam optimizer.
        #                 --adam_weight_decay ADAM_WEIGHT_DECAY
        #                                         Weight decay to use.
        #                 --adam_epsilon ADAM_EPSILON
        #                                         Epsilon value for the Adam optimizer
        #                 --max_grad_norm MAX_GRAD_NORM
        #                                         Max gradient norm.
        #                 --push_to_hub         Whether or not to push the model to the Hub.
        #                 --hub_token HUB_TOKEN
        #                                         The token to use to push to the Model Hub.
        #                 --hub_model_id HUB_MODEL_ID
        #                                         The name of the repository to keep in sync with the
        #                                         local `output_dir`.
        #                 --logging_dir LOGGING_DIR
        #                                         [TensorBoard](https://www.tensorflow.org/tensorboard)
        #                                         log directory. Will default to
        #                                         *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***.
        #                 --allow_tf32          Whether or not to allow TF32 on Ampere GPUs. Can be
        #                                         used to speed up training. For more information, see h
        #                                         ttps://pytorch.org/docs/stable/notes/cuda.html#tensorf
        #                                         loat-32-tf32-on-ampere-devices
        #                 --report_to REPORT_TO
        #                                         The integration to report the results and logs to.
        #                                         Supported platforms are `"tensorboard"` (default),
        #                                         `"wandb"` and `"comet_ml"`. Use `"all"` to report to
        #                                         all integrations.
        #                 --mixed_precision {no,fp16,bf16}
        #                                         Whether to use mixed precision. Choose between fp16
        #                                         and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10.and
        #                                         an Nvidia Ampere GPU. Default to the value of
        #                                         accelerate config of the current system or the flag
        #                                         passed with the `accelerate.launch` command. Use this
        #                                         argument to override the accelerate config.
        #                 --prior_generation_precision {no,fp32,fp16,bf16}
        #                                         Choose prior generation precision between fp32, fp16
        #                                         and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10.and
        #                                         an Nvidia Ampere GPU. Default to fp16 if a GPU is
        #                                         available else fp32.
        #                 --local_rank LOCAL_RANK
        #                                         For distributed training: local_rank
        #                 --enable_xformers_memory_efficient_attention
        #                 Whether or not to use xformers.
        #                 ```
        #                 """)
        #             train_lora_command = """python -u /content/trainer/diffusers/lora/train_dreambooth_lora.py \\
        #             --pretrained_model_name_or_path="/content/model"  \\
        #             --instance_data_dir="/content/drive/MyDrive/AI/training/parkminyoung" \\
        #             --output_dir="/content/trainer/diffusers/lora/output_dir" \\
        #             --learning_rate=5e-6 \\
        #             --max_train_steps=650 \\
        #             --instance_prompt="parkminyoung" \\
        #             --resolution=512 \\
        #             --center_crop \\
        #             --train_batch_size=1 \\
        #             --gradient_accumulation_steps=1 \\
        #             --max_grad_norm=1.0 \\
        #             --mixed_precision="fp16" \\
        #             --gradient_checkpointing \\
        #             --enable_xformers_memory_efficient_attention \\
        #             --use_8bit_adam"""
        #             lora_command = gr.Textbox(show_label=False, lines=16, value=train_lora_command)
        #             train_lora_out_text = gr.Textbox(show_label=False)
        #             btn_train_lora_run_live = gr.Button("Train Lora")
        #             btn_train_lora_run_live.click(run_live, inputs=lora_command, outputs=train_lora_out_text, show_progress=False)
        #     with gr.Tab("Test"):
        #         with gr.Group():
        #             with gr.Row():
        #                 with gr.Box():
        #                     image = gr.Image(show_label=False)
        #                 with gr.Box():
        #                     model_dir = gr.Textbox(label="Enter your output dir", show_label=False, max_lines=1, value="/content/model")
        #                     output_dir = gr.Textbox(label="Enter your output dir", show_label=False, max_lines=1, value="/content/trainer/diffusers/lora/output_dir")
        #                     prompt = gr.Textbox(label="prompt", show_label=False, max_lines=1, placeholder="Enter your prompt")
        #                     negative_prompt = gr.Textbox(label="negative prompt", show_label=False, max_lines=1, placeholder="Enter your negative prompt")
        #                     steps = gr.Slider(label="Steps", minimum=5, maximum=50, value=25, step=1)
        #                     scale = gr.Slider(label="Guidance Scale", minimum=0, maximum=50, value=7.5, step=0.1)
        #                     checkbox = gr.Checkbox(label="Load Model", value=True)
        #                     btn_test_lora = gr.Button("Generate image")
        #                     btn_test_lora.click(test_lora, inputs=[model_dir, checkbox, output_dir, prompt, negative_prompt, steps, scale], outputs=image) 
        #     with gr.Tab("Tools"):
        #         with gr.Group():
        #             with gr.Box():
        #                 with gr.Accordion("Copy Lora to Additional Network", open=False):
        #                     gr.Markdown(
        #                     """
        #                     ```py
        #                     cp /content/trainer/diffusers/lora/output_dir/pytorch_lora_weights.safetensors \\
        #                     /content/stable-diffusion-webui/extensions/sd-webui-additional-networks/models/lora/parkminyoung.safetensors
        #                     ```
        #                     """)
        #                 cp_lora_command = """cp /content/trainer/diffusers/lora/output_dir/pytorch_lora_weights.safetensors \\
        #                 /content/stable-diffusion-webui/extensions/sd-webui-additional-networks/models/lora/parkminyoung.safetensors"""
        #                 cp_lora = gr.Textbox(show_label=False, lines=2, value=cp_lora_command)
        #                 cp_lora_out_text = gr.Textbox(show_label=False)
        #                 btn_run_static = gr.Button("Copy Lora to Additional Network")
        #                 btn_run_static.click(run_live, inputs=cp_lora, outputs=cp_lora_out_text, show_progress=False)
        #         with gr.Group():
        #             with gr.Box():
        #                 with gr.Accordion("Remove Lora Output Directory", open=False):
        #                     gr.Markdown(
        #                     """
        #                     ```py
        #                     rm -rf /content/trainer/diffusers/lora/output_dir/*
        #                     ```
        #                     """)
        #                 rm_lora_command = """rm -rf /content/trainer/diffusers/lora/output_dir/*"""
        #                 rm_lora = gr.Textbox(show_label=False, lines=1, value=rm_lora_command)
        #                 rm_lora_out_text = gr.Textbox(show_label=False)
        #                 btn_run_static = gr.Button("Remove Lora Output Directory")
        #                 btn_run_static.click(run_live, inputs=rm_lora, outputs=rm_lora_out_text, show_progress=False)
        with gr.Tab("Textual Inversion for WebUI and Diffusers Lib"):
            with gr.Tab("Train"):
                with gr.Box():
                    with gr.Accordion("Train Textual Inversion Common Arguments", open=False):
                        gr.Markdown(
                        """
                        ```py
                        --pretrained_model_name_or_path="/content/model"  \\
                        --instance_data_dir="/content/drive/MyDrive/AI/training/parkminyoung" \\
                        --output_dir="/content/trainer/diffusers/dreambooth/output_dir" \\
                        --learning_rate=5e-6 \\
                        --max_train_steps=650 \\
                        --instance_prompt="parkminyoung" \\
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
                        --only_save_embeds    Save only the embeddings for the new concept.
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
                        --validation_epochs VALIDATION_EPOCHS
                                                Run validation every X epochs. Validation consists of
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
                                                Max number of checkpoints to store. Passed as
                                                `total_limit` to the `Accelerator`
                                                `ProjectConfiguration`. See Accelerator::save_state ht
                                                tps://huggingface.co/docs/accelerate/package_reference
                                                /accelerator#accelerate.Accelerator.save_state for
                                                more docs
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
                    --train_data_dir="/content/drive/MyDrive/AI/training/parkminyoung" \\
                    --learnable_property="object" \\
                    --output_dir="/content/trainer/diffusers/textual_inversion/output_dir" \\
                    --placeholder_token="<parkminyoung>" \\
                    --initializer_token="parkminyoung" \\
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
                    btn_train_textual_inversion_run_live.click(run_live, inputs=textual_inversion_command, outputs=train_textual_inversion_out_text, show_progress=False)
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
                            btn_test_dreambooth.click(test_dreambooth, inputs=[output_dir, checkbox, prompt, negative_prompt, steps, scale], outputs=image)
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
                        btn_run_static.click(run_live, inputs=rm_dreambooth, outputs=rm_dreambooth_out_text, show_progress=False)
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
                    btn_train_tag_lora_webui_run_live.click(run_live, inputs=tag_lora_webui_command, outputs=train_tag_lora_webui_out_text, show_progress=False)
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
                    btn_train_merge_lora_webui_run_live.click(run_live, inputs=merge_lora_webui_command, outputs=train_merge_lora_webui_out_text, show_progress=False)
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
                    btn_train_prepare_lora_webui_run_live.click(run_live, inputs=prepare_lora_webui_command, outputs=train_prepare_lora_webui_out_text, show_progress=False)
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
                        btn_train_lora_webui_run_live.click(run_live, inputs=lora_webui_command, outputs=train_lora_webui_out_text, show_progress=False)
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
                        btn_run_static.click(run_live, inputs=rm_lora, outputs=rm_lora_out_text, show_progress=False)
    trainer.queue().launch(debug=True, share=True, inline=False)

if __name__ == "__main__":
    launch()
