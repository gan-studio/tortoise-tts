"""
ModelHandler defines a custom model handler.
"""
import base64
import io
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import boto3
import torch
import torchaudio
from ts.torch_handler.base_handler import BaseHandler

s3_client = boto3.client(
    service_name="s3",
    region_name=os.getenv("AWS_DEFAULT_REGION"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)


def is_folder_empty(folder_path: str):
    if not os.path.exists(folder_path):
        return True

    return not any(os.listdir(folder_path))


def download_file(bucket_name: str, key: str, download_path: str):
    try:
        filename = key.split("/")[-1]
        if ".wav" in filename:
            s3_client.download_file(
                bucket_name, key, os.path.join(download_path, filename)
            )
    except Exception as error:
        print(f"Error downloading {key}: {error}")


def download_all_files_parallel(bucket_name: str, folder_key: str, download_path: str):
    os.makedirs(download_path, exist_ok=True)
    objects = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=folder_key)
    keys = [obj["Key"] for obj in objects.get("Contents", [])]

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(download_file, bucket_name, key, download_path)
            for key in keys
        ]
        for future in futures:
            future.result()

    if is_folder_empty(download_path):
        print(f"No sample audio found for {folder_key}")
        raise Exception(f"No sample audio found {folder_key}")


class ModelHandler(BaseHandler):
    """
    A custom model handler implementation.
    """

    def __init__(self):
        self._context = None
        self.initialized = False

    def initialize(self, context):
        """
        Initialize model. This will be called during model loading time
        :param context: Initial context contains model server system properties.
        :return:
        """
        self._context = context
        self.manifest = context.manifest
        properties = context.system_properties

        model_dir = properties.get("model_dir")

        checkpoint_path = os.path.join(model_dir, "tts_model.pth")

        model_name = model_dir.split("/")[-1]
        temp_path = f"/tmp/{model_name}"
        os.makedirs(temp_path, exist_ok=True)
        shutil.unpack_archive(os.path.join(model_dir, "tts.zip"), temp_path)
        sys.path.append(os.path.join(temp_path))
        sys.path.append(os.path.join(temp_path, "tortoise"))
        self.temp_tortoise_path = os.path.join(temp_path, "tortoise")
        from tortoise.combined_model import TTSModel

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TTSModel(
            autoregressive_batch_size=16,
            use_deepspeed=False,
            kv_cache=True,
            device=self.device,
        )
        self.model.from_pretrained(checkpoint_path)

        self.initialized = True

    def preprocess(self, data: list):
        """
        Transform raw input into model input data.
        :param batch: list of raw requests, should match batch size
        :return: list of preprocessed model input data
        """
        # Take the input data and make it inference ready
        print(f"------Preprocess------")
        print(f"Batch Size: {len(data)}")
        start_time = time.time()
        preprocessed_data = data[0].get("data")
        if preprocessed_data is None:
            preprocessed_data = json.loads(data[0].get("body").decode("utf-8"))

        (text, voice, voice_path_s3, bucket_name, preset) = (
            preprocessed_data.get("text"),
            preprocessed_data.get("voice"),
            preprocessed_data.get("voice_path_s3"),
            preprocessed_data.get("s3_bucket_name"),
            preprocessed_data.get("preset", "fast"),
        )
        local_voice_path = os.path.join(self.temp_tortoise_path, "voices", voice)
        download_all_files_parallel(bucket_name, voice_path_s3, local_voice_path)
        from tortoise.utils.audio import load_voice

        voice_samples, conditioning_latents = load_voice(voice)

        print(f"Total Time Taken for Pre Process: {time.time()-start_time} seconds")
        return [
            {
                "text": text,
                "voice_samples": voice_samples,
                "conditioning_latents": conditioning_latents,
                "preset": preset,
            }
        ]

    def inference(self, model_input: list):
        """
        Internal inference methods
        :param model_input: transformed model input data
        :return: list of inference output in NDArray
        """
        # Do some inference call to engine here and return output
        print(f"------Inference------")
        start_time = time.time()
        text, voice_samples, conditioning_latents, preset = (
            model_input[0]["text"],
            model_input[0]["voice_samples"],
            model_input[0]["conditioning_latents"],
            model_input[0]["preset"],
        )
        model_output = self.model.tts_with_preset(
            text=text,
            voice_samples=voice_samples,
            conditioning_latents=conditioning_latents,
            preset=preset,
        )
        model_output = [model_output.squeeze(0).cpu().numpy()]
        print(f"Total Time Taken Inference: {time.time()-start_time} seconds")
        return model_output

    def postprocess(self, inference_output: list):
        """
        Return inference result.
        :param inference_output: list of inference output
        :return: list of predict results
        """
        # Take output from network and post-process to desired format

        print("------Post Process------")
        start_time = time.time()
        wavIO = io.BytesIO()
        torchaudio.save(
            wavIO, torch.from_numpy(inference_output[0]), 24000, format="wav"
        )
        encoded_audio = base64.b64encode(wavIO.getvalue()).decode("utf-8")
        output = [{"wav": encoded_audio}]
        print(f"Total Time Taken for Post Process: {time.time()-start_time} seconds")
        return output

    def handle(self, data, context):
        """
        Invoke by TorchServe for prediction request.
        Do pre-processing of data, prediction using model and postprocessing of prediction output
        :param data: Input data for prediction
        :param context: Initial context contains model server system properties.
        :return: prediction output
        """
        if not self.initialized:
            self.initialized(context)

        model_input = self.preprocess(data)
        model_output = self.inference(model_input)
        return self.postprocess(model_output)
