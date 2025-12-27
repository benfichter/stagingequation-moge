from __future__ import annotations

import base64

from inference import infer_dimensions, warm_model


def _decode_base64_image(payload: str) -> bytes:
    if "," in payload:
        payload = payload.split(",", 1)[1]
    return base64.b64decode(payload)


def handler(job: dict) -> dict:
    input_data = job.get("input", {}) if isinstance(job, dict) else {}

    if input_data.get("warm"):
        warm_model()
        return {"status": "warmed"}

    image_base64 = input_data.get("image_base64")
    if not image_base64:
        return {"error": "image_base64 is required"}

    calibration_height = input_data.get("calibration_height_m")
    if calibration_height is not None:
        try:
            calibration_height = float(calibration_height)
        except (TypeError, ValueError):
            calibration_height = None

    try:
        image_bytes = _decode_base64_image(image_base64)
    except Exception:
        return {"error": "invalid image_base64"}

    return infer_dimensions(image_bytes, calibration_height)
