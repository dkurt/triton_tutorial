# -*- coding: utf-8 -*-
import os
from pathlib import Path
from typing import Any, Generator

import numpy as np
import pytest
from tritonclient.http import InferenceServerClient

SERVER_URL = os.environ.get("SERVER_URL", "0.0.0.0:18000")


@pytest.fixture(scope="session")
def client() -> Generator[InferenceServerClient, Any, None]:
    cls = InferenceServerClient(
        url=SERVER_URL,
        network_timeout=None,
        connection_timeout=None,
    )
    if not cls.is_server_ready():
        pytest.fail(f"Triton server {SERVER_URL} reported not ready")
    yield cls


@pytest.fixture(scope="session")
def car_video() -> np.ndarray:
    video_path = Path(os.environ["VIDEO_PATH"])
    return np.fromfile(video_path, dtype=np.uint8).reshape(1, -1)
