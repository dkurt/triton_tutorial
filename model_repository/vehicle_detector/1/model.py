import json
from typing import Dict, List

import numpy as np
import triton_python_backend_utils as pb_utils

DEFAULT_THRESHOLD = 0.5

VEHICLE_LABELS = {
    "bus",
    "car",
    "truck",
}


class TritonPythonModel:
    def initialize(self, args: Dict[str, str]) -> None:
        from transformers import AutoImageProcessor, RfDetrForObjectDetection

        import torch

        device_kind = "cuda" if args["model_instance_kind"] == "GPU" else "cpu"
        device_id = args["model_instance_device_id"]
        self.device = f"{device_kind}:{device_id}"

        # Fall back to CPU when CUDA isn't actually available (e.g. the
        # container was started without GPU passthrough).
        if device_kind == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"

        self.processor = AutoImageProcessor.from_pretrained(
            "stevenbucaille/rf-detr-medium"
        )
        self.model = RfDetrForObjectDetection.from_pretrained(
            "stevenbucaille/rf-detr-medium"
        ).to(self.device)
        self.model.eval()

        # Label id -> human name for the vehicle classes of interest.
        id2label = self.model.config.id2label
        self.vehicle_id2name = {
            idx: name for idx, name in id2label.items() if name in VEHICLE_LABELS
        }

        print(f"[vehicle_detector] initialized on {self.device}", flush=True)

    def execute(
        self, requests: List["pb_utils.InferenceRequest"]
    ) -> List["pb_utils.InferenceResponse"]:
        import torch

        responses = []
        for request in requests:
            # Confidence threshold comes from the caller (request parameter),
            # defaulting to DEFAULT_THRESHOLD when not supplied.
            threshold = DEFAULT_THRESHOLD
            raw_params = request.parameters()
            if raw_params:
                try:
                    params = json.loads(raw_params)
                    threshold = float(params.get("threshold", DEFAULT_THRESHOLD))
                except (ValueError, TypeError):
                    threshold = DEFAULT_THRESHOLD

            frame = pb_utils.get_input_tensor_by_name(request, "image").as_numpy()[0]

            inputs = self.processor(images=frame, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)

            # convert outputs (bounding boxes and class logits) to COCO API and
            # keep only detections above the requested threshold. target_sizes is
            # [[height, width]] in image pixel coordinates.
            target_sizes = torch.tensor([frame.shape[:2]], dtype=torch.int32)
            results = self.processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=threshold
            )[0]

            boxes: list[list[float]] = []

            for score, label, box in zip(
                results["scores"], results["labels"], results["boxes"]
            ):
                name = self.vehicle_id2name.get(int(label.item()))
                if name is None:
                    continue
                boxes.append([round(float(v), 2) for v in box.tolist()])

            if not boxes:
                boxes = [[0.0, 0.0, 0.0, 0.0]]

            responses.append(
                pb_utils.InferenceResponse(
                    output_tensors=[
                        pb_utils.Tensor("boxes", np.asarray(boxes, dtype=np.float32)),
                    ]
                )
            )
        return responses
