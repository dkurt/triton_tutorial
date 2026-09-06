# -*- coding: utf-8 -*-
import numpy as np
import json
import pytest
from tritonclient.http import InferInput, InferRequestedOutput

MODEL_NAME: str = "video_ocr"
INPUT_NAME: str = "video"
OUTPUT_NAME: str = "names"

EXPECTED_OUTPUT: list[str] = [
    "H332PM152",
    "X090AP252",
    "B376HB92",
    "H269YA152",
]


class TestVideoOcr:
    @pytest.mark.timeout(1200)
    def test_video_ocr_expected_texts(
        self, client, car_video: np.ndarray
    ) -> None:
        video_in = InferInput(INPUT_NAME, list(car_video.shape), "UINT8")
        video_in.set_data_from_numpy(car_video)

        output = client.infer(
            MODEL_NAME,
            inputs=[video_in],
            outputs=[InferRequestedOutput(OUTPUT_NAME)],
            parameters={"skip_every_frame": "4"}
        )

        results = json.loads(output.as_numpy(OUTPUT_NAME)[0])

        assert all(number in results for number in EXPECTED_OUTPUT)
