#!/bin/bash

mkdir -p model_analyzer/results

docker run  --gpus all --net=host \
    -v ./model_repository:/model_repository \
    -v ./model_analyzer:/model_analyzer \
    nvcr.io/nvidia/tritonserver:24.12-py3-sdk \
    model-analyzer profile \
    -f /model_analyzer/config.yml \
    --model-repository /model_repository \
    --profile-models video_ocr \
    --triton-launch-mode=remote \
    --triton-http-endpoint=0.0.0.0:18000 \
    --triton-grpc-endpoint=0.0.0.0:18001 \
    --triton-metrics-url=http://0.0.0.0:18002/metrics \
    --output-model-repository-path /model_analyzer/results/video_ocr \
    --override-output-model-repository \
    --export-path profile_results \
    --run-config-search-max-model-batch-size 1
