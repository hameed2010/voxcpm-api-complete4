FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y python3 python3-pip ffmpeg git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir --upgrade pip
RUN pip3 install --no-cache-dir --index-url https://download.pytorch.org/whl/cu124 torch torchaudio
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

# RunPod Serverless: run the handler (not uvicorn)
CMD ["python3", "-u", "handler.py"]
