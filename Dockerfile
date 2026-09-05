FROM nvcr.io/nvidia/tritonserver:24.12-py3

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock /opt/triton-demo-src/
RUN pip install --break-system-packages uv \
    && uv venv /opt/triton-demo/venv \
    && cd /opt/triton-demo-src \
    && VIRTUAL_ENV=/opt/triton-demo/venv uv sync --active --frozen

# Pre-stage the easyocr weights so the model loads offline at startup instead
# of block-downloading them over the network (which hangs server readiness).
RUN mkdir -p /opt/easyocr-models \
    && . /opt/triton-demo/venv/bin/activate \
    && python -c "import easyocr; easyocr.Reader(['ru'], gpu=False, detect_network='craft', recog_network='cyrillic_g2', download_enabled=True, model_storage_directory='/opt/easyocr-models')"

COPY model_repository /model_repository
