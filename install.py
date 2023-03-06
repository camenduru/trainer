import os, launch
from modules import scripts

sd_scripts_repo_commit_hash = os.environ.get('SD_SCRIPTS_COMMIT_HASH', "772ee52ef2f0ccdb56f9922ab1bf03a9c0308697")
sd_scripts_repo = os.environ.get('SD_SCRIPTS_REPO', "https://github.com/camenduru/sd-scripts")
sd_scripts_dir = os.path.join(scripts.basedir(), "extensions", "stable-diffusion-webui-trainer", "sd-scripts")
launch.git_clone(sd_scripts_repo, sd_scripts_dir, "SD Scripts", sd_scripts_repo_commit_hash)

sd_scripts_requirements_file = os.environ.get('SD_SCRIPTS_REQS_FILE', os.path.join(sd_scripts_dir, "requirements.txt"))
launch.run_pip(f"install -r {sd_scripts_requirements_file}", "requirements for sd scripts")

launch.run_pip("install diffusers==0.13.1", "diffusers==0.13.1 requirements for trainer extension")
launch.run_pip("install transformers==4.26.1", "transformers==4.26.1 requirements for trainer extension")
# launch.run_pip("install ftfy==6.1.1", "ftfy==6.1.1 requirements for trainer extension")
# launch.run_pip("install accelerate==0.16.0", "accelerate==0.16.0 requirements for trainer extension")
# launch.run_pip("install bitsandbytes==0.37.0", "bitsandbytes==0.37.0 requirements for trainer extension")
# launch.run_pip("install safetensors==0.2.8", "safetensors==0.2.8 requirements for trainer extension")

# launch.run_pip(f"install {sd_scripts_dir}", "kohya_ss library requirements for trainer extension")