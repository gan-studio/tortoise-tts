import os
import subprocess
import sys
import zipfile

import boto3
import torch
import yaml

print(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "tortoise"))


import os


def create_zip(output_dir: str):
    zip_path = os.path.join(output_dir, "tts.zip")
    zf = zipfile.ZipFile(zip_path, "w")
    for dirname, subdirs, files in os.walk("tortoise"):
        if (
            "voices" in dirname
            or "__pycache__" in dirname
            or ".ipynb_checkpoints" in dirname
        ):
            continue
        for filename in files:
            if (
                ".pyc" not in filename
                and ".ipynb" not in filename
                and ".pth" not in filename
                and "handler" not in filename
                and "LICENSE" not in filename
                and "MANIFEST.json" not in filename
                and ".ipynb_checkpoints" not in filename,
            ):
                zf.write(os.path.join(dirname, filename))
    return zip_path


def upload_to_aws(bucket: str, local_file: str, s3_file: str):
    s3 = None
    try:
        AWS_ACCESS_KEY_ID = os.environ["AWS_ACCESS_KEY"]
        AWS_SECRET_ACCESS_KEY = os.environ["AWS_SECRET_ACCESS_KEY"]
        s3 = boto3.resource(
            "s3",
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name="ap-south-1",
        )
    except:
        s3 = boto3.resource("s3")
    s3.Bucket(bucket).upload_file(local_file, s3_file)


def create_combined_model(
    autoregressive_ckpt_path: str,
    clvp2_ckpt_path: str,
    diffusion_decoder_ckpt_path: str,
    vocoder_ckpt_path: str,
    rlg_auto_ckpt_path: str,
    rlg_diffuser_ckpt_path: str,
    cvvp_ckpt_path: str,
    output_dir: str,
):
    autoregressive_ckpt = torch.load(autoregressive_ckpt_path)
    clvp2_ckpt = torch.load(clvp2_ckpt_path)
    diffusion_decoder_ckpt = torch.load(diffusion_decoder_ckpt_path)
    vocoder_ckpt = torch.load(vocoder_ckpt_path, map_location="cpu")
    rlg_auto_ckpt = torch.load(rlg_auto_ckpt_path)
    rlg_diffuser_ckpt = torch.load(rlg_diffuser_ckpt_path)
    cvvp_ckpt = torch.load(cvvp_ckpt_path)
    output_path = os.path.join(output_dir, "tts_model.pth")
    torch.save(
        {
            "autoregressive": autoregressive_ckpt,
            "clvp2": clvp2_ckpt,
            "diffusion_decoder": diffusion_decoder_ckpt,
            "generator": vocoder_ckpt,
            "rlg_auto": rlg_auto_ckpt,
            "rlg_diffuser": rlg_diffuser_ckpt,
            "cvvp": cvvp_ckpt,
        },
        output_path,
    )

    return output_path


def create_torchserve_archive(model_name: str, model_ckpt_path: str, extra_files: str):
    command = [
        "torch-model-archiver",
        "--model-name",
        model_name,
        "--version",
        "1.0",
        "--model-file",
        "tortoise/combined_model.py",
        "--serialized-file",
        model_ckpt_path,
        "--handle",
        "torchserve_handler.py",
        "--archive-format",
        "tgz",
        "--config-file",
        "torch_serve_model_config.yaml",
        "--extra-files",
        extra_files,
    ]

    # Run the command
    subprocess.run(command, check=True)
    return f"{model_name}.tar.gz"


def create_torchserve_model(
    sagemaker_bucket: str,
    autoregressive_ckpt_path: str,
    clvp2_ckpt_path: str,
    diffusion_decoder_ckpt_path: str,
    vocoder_ckpt_path: str,
    rlg_auto_ckpt_path: str,
    rlg_diffuser_ckpt_path: str,
    cvvp_ckpt_path: str,
    output_dir: str,
    training_job_id: str,
):
    zip_path = create_zip(output_dir)
    model_ckpt_path = create_combined_model(
        autoregressive_ckpt_path,
        clvp2_ckpt_path,
        diffusion_decoder_ckpt_path,
        vocoder_ckpt_path,
        rlg_auto_ckpt_path,
        rlg_diffuser_ckpt_path,
        cvvp_ckpt_path,
        output_dir,
    )
    extra_files = f"{zip_path}"

    model_path = create_torchserve_archive(
        training_job_id, model_ckpt_path, extra_files
    )
    upload_to_aws(
        sagemaker_bucket,
        model_path,
        f"torchserve-tortoise/tortoise-models/{model_path}",
    )
