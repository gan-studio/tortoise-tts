ARG BASE_IMAGE

FROM $BASE_IMAGE

RUN pip install tqdm
RUN pip install rotary_embedding_torch
RUN pip install transformers==4.31.0
RUN pip install tokenizers
RUN pip install inflect
RUN pip install progressbar
RUN pip install einops==0.4.1
RUN pip install unidecode
RUN pip install scipy
RUN pip install librosa==0.9.1
RUN pip install ffmpeg
RUN pip install numpy
RUN pip install numba
RUN pip install torchaudio
RUN pip install threadpoolctl
RUN pip install llvmlite
RUN pip install appdirs
RUN pip install nbconvert==5.3.1
RUN pip install tornado==4.2
RUN pip install pydantic==1.9.1
RUN pip install deepspeed==0.8.3
RUN pip install py-cpuinfo
RUN pip install hjson
RUN pip install psutil
RUN pip install sounddevice
RUN pip install boto3==1.28.12

ENV S3_AWS_REGION_NAME=manash
ENV S3_AWS_ACCESS_KEY_ID=manash
ENV S3_AWS_SECRET_ACCESS_KEY=manash
ENV MODEL_SERVER_TIMEOUT=100000