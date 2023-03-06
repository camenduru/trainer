import os

# from https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/master/launch.py

def run(command, desc=None, errdesc=None, custom_env=None, live=False):
    if desc is not None:
        print(desc)

    if live:
        result = subprocess.run(command, shell=True, env=os.environ if custom_env is None else custom_env)
        if result.returncode != 0:
            raise RuntimeError(f"""{errdesc or 'Error running command'}.
Command: {command}
Error code: {result.returncode}""")

        return ""

    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, env=os.environ if custom_env is None else custom_env)

    if result.returncode != 0:

        message = f"""{errdesc or 'Error running command'}.
Command: {command}
Error code: {result.returncode}
stdout: {result.stdout.decode(encoding="utf8", errors="ignore") if len(result.stdout)>0 else '<empty>'}
stderr: {result.stderr.decode(encoding="utf8", errors="ignore") if len(result.stderr)>0 else '<empty>'}
"""
        raise RuntimeError(message)

    return result.stdout.decode(encoding="utf8", errors="ignore")


def git_clone(url, dir, name, commithash=None):
    # TODO clone into temporary dir and move if successful

    if os.path.exists(dir):
        if commithash is None:
            return

        current_hash = run(f'"{git}" -C "{dir}" rev-parse HEAD', None, f"Couldn't determine {name}'s hash: {commithash}").strip()
        if current_hash == commithash:
            return

        run(f'"{git}" -C "{dir}" fetch', f"Fetching updates for {name}...", f"Couldn't fetch {name}")
        run(f'"{git}" -C "{dir}" checkout {commithash}', f"Checking out commit for {name} with hash: {commithash}...", f"Couldn't checkout commit {commithash} for {name}")
        return

    run(f'"{git}" clone "{url}" "{dir}"', f"Cloning {name} into {dir}...", f"Couldn't clone {name}")

    if commithash is not None:
        run(f'"{git}" -C "{dir}" checkout {commithash}', None, "Couldn't checkout {name}'s hash: {commithash}")


def run_pip(args, desc=None):
    if skip_install:
        return

    index_url_line = f' --index-url {index_url}' if index_url != '' else ''
    return run(f'"{python}" -m pip {args} --prefer-binary{index_url_line}', desc=f"Installing {desc}", errdesc=f"Couldn't install {desc}")


def install():
    sd_scripts_repo_commit_hash = os.environ.get('SD_SCRIPTS_COMMIT_HASH', "772ee52ef2f0ccdb56f9922ab1bf03a9c0308697")
    sd_scripts_repo = os.environ.get('SD_SCRIPTS_REPO', "https://github.com/camenduru/sd-scripts")
    sd_scripts_dir = os.path.join(os.getcwd(), "sd-scripts")
    git_clone(sd_scripts_repo, sd_scripts_dir, "SD Scripts", sd_scripts_repo_commit_hash)

    sd_scripts_requirements_file = os.environ.get('SD_SCRIPTS_REQS_FILE', os.path.join(sd_scripts_dir, "requirements.txt"))
    run_pip(f"install -r {sd_scripts_requirements_file}", "requirements for sd scripts")

    run_pip("install diffusers==0.13.1", "diffusers==0.13.1 requirements for trainer extension")
    run_pip("install transformers==4.26.1", "transformers==4.26.1 requirements for trainer extension")
    # run_pip("install ftfy==6.1.1", "ftfy==6.1.1 requirements for trainer extension")
    # run_pip("install accelerate==0.16.0", "accelerate==0.16.0 requirements for trainer extension")
    # run_pip("install bitsandbytes==0.37.0", "bitsandbytes==0.37.0 requirements for trainer extension")
    # run_pip("install safetensors==0.2.8", "safetensors==0.2.8 requirements for trainer extension")

    # run_pip(f"install {sd_scripts_dir}", "kohya_ss library requirements for trainer extension")

if __name__ == "__main__":
    install()
    from scripts import trainer
    trainer.launch()