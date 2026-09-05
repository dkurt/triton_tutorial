import json
from typing import Dict, List

import numpy as np
import triton_python_backend_utils as pb_utils

DEFAULT_MIN_CONF = 0.5


class TritonPythonModel:
    def initialize(self, args: Dict[str, str]) -> None:
        import torch

        import easyocr

        device_kind = "cuda" if args["model_instance_kind"] == "GPU" else "cpu"
        device_id = args["model_instance_device_id"]
        self.device = f"{device_kind}:{device_id}"

        # Fall back to CPU when CUDA isn't actually available (e.g. the
        # container was started without GPU passthrough).
        if device_kind == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"

        # Public pretrained models: english + cyrillic (ru). Downloaded on demand.
        self.reader = easyocr.Reader(
            ["ru"],
            gpu=False if self.device.startswith("cpu") else self.device,
            detect_network="craft",
            recog_network="cyrillic_g2",
            download_enabled=True,
        )

        chars = self.reader.lang_char
        whitelist = (
            "АВЕКМНОРСТУХ0123456789"
        )
        self.allowlist = "".join(sorted(set(chars).intersection(set(whitelist))))

        print(f"[easyocr] initialized on {self.device}", flush=True)

    def execute(
        self, requests: List["pb_utils.InferenceRequest"]
    ) -> List["pb_utils.InferenceResponse"]:
        responses = []
        for request in requests:
            # Confidence threshold comes from the caller (request parameter),
            # defaulting to DEFAULT_MIN_CONF when not supplied.
            min_conf = DEFAULT_MIN_CONF
            raw_params = request.parameters()
            if raw_params:
                try:
                    params = json.loads(raw_params)
                    min_conf = float(params.get("min_conf", DEFAULT_MIN_CONF))
                except (ValueError, TypeError):
                    min_conf = DEFAULT_MIN_CONF

            frame = pb_utils.get_input_tensor_by_name(request, "image").as_numpy()[0]

            recognized = self.reader.readtext(frame, allowlist=self.allowlist)

            texts: list[bytes] = []

            for bbox, text, conf in recognized:
                if conf < min_conf or not text:
                    continue
                texts.append(text.encode("utf-8"))

            if not texts:
                texts = [b""]

            responses.append(
                pb_utils.InferenceResponse(
                    output_tensors=[
                        # Triton STRING needs a fixed-width numpy "S" dtype sized
                        # to the longest kept text.
                        pb_utils.Tensor(
                            "text", np.asarray(texts, dtype=f"S{max(len(n) for n in texts)}")
                        ),
                    ]
                )
            )
        return responses
