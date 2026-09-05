FROM nvcr.io/nvidia/tritonserver:24.12-py3

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock /opt/triton-demo-src/
RUN pip install --break-system-packages uv \
    && uv venv /opt/triton-demo/venv \
    && cd /opt/triton-demo-src \
    && VIRTUAL_ENV=/opt/triton-demo/venv uv sync --active --frozen

COPY model_repository /model_repository
