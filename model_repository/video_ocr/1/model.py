import asyncio
import tempfile
from typing import List
import json

import numpy as np
import cv2 as cv
import triton_python_backend_utils as pb_utils

MIN_NUM_CHARACTERS = 8

# The deployed models this orchestrator calls. Referenced by model name so
# Triton resolves them internally (no host/port).
VEHICLE_MODEL = "vehicle_detector"
OCR_MODEL = "easyocr"


class TritonPythonModel:
    async def _detect_vehicles(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        batch = frame.astype(np.uint8)[None]  # [1, H, W, C]
        request = pb_utils.InferenceRequest(
            model_name=VEHICLE_MODEL,
            inputs=[pb_utils.Tensor("image", batch)],
            requested_output_names=["boxes"],
        )
        response = await request.async_exec()
        if response.has_error():
            raise pb_utils.TritonModelException(response.error().message())

        boxes_tensor = pb_utils.get_output_tensor_by_name(response, "boxes")
        if boxes_tensor is None or boxes_tensor.as_numpy() is None:
            return []
        boxes = boxes_tensor.as_numpy()  # [N, 4], float32 xyxy

        h, w = frame.shape[:2]
        out: list[tuple[int, int, int, int]] = []
        for row in boxes:
            x1, y1, x2, y2 = (int(round(float(v))) for v in row[:4])
            if x2 <= x1 or y2 <= y1:
                continue
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            out.append((x1, y1, x2, y2))
        return out

    async def _detect_text(self, crop: np.ndarray) -> list[bytes]:
        request = pb_utils.InferenceRequest(
            model_name=OCR_MODEL,
            inputs=[pb_utils.Tensor("image", np.expand_dims(crop, axis=0))],
            requested_output_names=["text"],
        )
        response = await request.async_exec()
        if response.has_error():
            raise pb_utils.TritonModelException(response.error().message())

        text_tensor = pb_utils.get_output_tensor_by_name(response, "text").as_numpy()

        out = []
        for text in text_tensor:
            if len(text.decode("utf-8")) >= MIN_NUM_CHARACTERS:
                out.append(text)
        return out

    async def _process_frame(self, frame: np.ndarray) -> list[bytes]:
        boxes = await self._detect_vehicles(frame)
        if not boxes:
            return []
        crops = [frame[y1:y2, x1:x2] for (x1, y1, x2, y2) in boxes]
        ocr_tasks = [asyncio.create_task(self._detect_text(crop)) for crop in crops]
        flattened = await asyncio.gather(*ocr_tasks)
        return [text for ocr_result in flattened for text in ocr_result]

    async def execute(
        self, requests: List["pb_utils.InferenceRequest"]
    ) -> List["pb_utils.InferenceResponse"]:
        responses = []
        print(f"[video_ocr] Processing {len(requests)} requests", flush=True)
        for request in requests:
            video_bytes = pb_utils.get_input_tensor_by_name(request, "video").as_numpy().tobytes()

            params = json.loads(request.parameters())
            skip_every_frame = int(params.get("skip_every_frame", "0"))

            names = await self._process_video(video_bytes, skip_every_frame)

            responses.append(
                pb_utils.InferenceResponse(
                    output_tensors=[
                        pb_utils.Tensor("names", names),
                    ]
                )
            )
        return responses

    async def _process_video(self, raw_video: bytes, skip_every_frame: int) -> np.ndarray:
        """Decode the video, run vehicle detection + OCR on each frame, keep
        the unique set of recognized texts."""
        tasks = []

        with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
            tmp.write(raw_video)
            tmp.flush()
            cap = cv.VideoCapture(tmp.name, cv.CAP_FFMPEG)
            if not cap.isOpened():
                raise RuntimeError("server failed to decode the uploaded video")
            frame_idx = 0
            while True:
                ok, frame = cap.read()
                frame_idx += 1
                if not ok:
                    break
                if skip_every_frame and frame_idx % skip_every_frame == 0:
                    continue
                tasks.append(asyncio.create_task(self._process_frame(frame)))
                # Let async tasks run
                if len(tasks) % 4 == 0:
                    await asyncio.sleep(0)
            cap.release()

        results = await asyncio.gather(*tasks)

        plates = set()
        for result in results:
            for text in result:
                text = list(text.decode("utf-8").replace("[", "").replace("]", "").upper())
                for i in [1, 2, 3]:
                    if text[i] == "O":
                        text[i] = "0"
                    elif text[i] == "B":
                        text[i] = "8"
                for i in [0, 4, 5]:
                    if text[i] == "4":
                        text[i] = "A"
                    elif text[i] == "8":
                        text[i] = "B"
                plates.add("".join(text))

        return np.asarray([json.dumps(list(plates)).encode("utf-8")])
