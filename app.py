from __future__ import annotations

import os
from typing import Any

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, status
from moge.model.v2 import MoGeModel

app = FastAPI(title="MoGe Dimensions API")

_MODEL: MoGeModel | None = None
_DEVICE: torch.device | None = None


def _require_api_key(request: Request) -> None:
    expected = os.getenv("MOGE_API_KEY")
    if not expected:
        return
    auth = request.headers.get("authorization", "")
    if auth != f"Bearer {expected}":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid api key")


def _resolve_device() -> torch.device:
    preferred = os.getenv("MOGE_DEVICE", "cuda").lower()
    if preferred == "cuda" and not torch.cuda.is_available():
        preferred = "cpu"
    return torch.device(preferred)


def _get_model() -> tuple[MoGeModel, torch.device]:
    global _MODEL
    global _DEVICE

    if _MODEL is not None and _DEVICE is not None:
        return _MODEL, _DEVICE

    _DEVICE = _resolve_device()
    model_id = os.getenv("MOGE_MODEL_ID", "Ruicheng/moge-2-vitl-normal")
    _MODEL = MoGeModel.from_pretrained(model_id).to(_DEVICE)
    _MODEL.eval()
    return _MODEL, _DEVICE


def _decode_image(image_bytes: bytes) -> np.ndarray:
    raw = np.frombuffer(image_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Invalid image data.")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _estimate_height(
    points: np.ndarray, mask: np.ndarray, normal: np.ndarray | None
) -> tuple[float, float, float] | None:
    valid_mask = mask > 0
    valid_points = points[valid_mask]
    if valid_points.size == 0:
        return None

    y_coords = valid_points[:, 1]
    floor_y = float(np.percentile(y_coords, 95))
    ceiling_y = float(np.percentile(y_coords, 5))

    if normal is not None:
        normal_y = normal[:, :, 1]
        floor_mask = (normal_y < -0.7) & valid_mask
        ceiling_mask = (normal_y > 0.7) & valid_mask
        if floor_mask.any():
            floor_y = float(np.mean(points[floor_mask][:, 1]))
        if ceiling_mask.any():
            ceiling_y = float(np.mean(points[ceiling_mask][:, 1]))

    height = abs(floor_y - ceiling_y)
    return height, floor_y, ceiling_y


def _floor_points(points: np.ndarray, mask: np.ndarray, normal: np.ndarray | None) -> np.ndarray | None:
    valid_mask = mask > 0
    valid_points = points[valid_mask]
    if valid_points.size == 0:
        return None

    if normal is not None:
        normal_y = normal[:, :, 1]
        floor_mask = (normal_y < -0.7) & valid_mask
        if floor_mask.any():
            return points[floor_mask]

    y_coords = valid_points[:, 1]
    threshold = np.percentile(y_coords, 80)
    floor_points = valid_points[valid_points[:, 1] > threshold]
    return floor_points if floor_points.size else valid_points


def _infer_dimensions(
    image_bytes: bytes, calibration_height_m: float | None
) -> dict[str, Any]:
    model, device = _get_model()
    image = _decode_image(image_bytes)
    tensor = torch.from_numpy(image).to(device).float() / 255.0
    tensor = tensor.permute(2, 0, 1)

    with torch.no_grad():
        output = model.infer(tensor)

    points = output["points"].detach().cpu().numpy()
    mask = output["mask"].detach().cpu().numpy()
    normal = output.get("normal")
    normal_map = normal.detach().cpu().numpy() if normal is not None else None

    height_data = _estimate_height(points, mask, normal_map)
    if height_data is None:
        raise RuntimeError("Unable to estimate room height.")
    height, _, _ = height_data

    floor_points = _floor_points(points, mask, normal_map)
    if floor_points is None:
        raise RuntimeError("Unable to determine floor points.")

    x_coords = floor_points[:, 0]
    z_coords = floor_points[:, 2]
    width = float(np.max(x_coords) - np.min(x_coords))
    depth = float(np.max(z_coords) - np.min(z_coords))
    area = width * depth

    calibration_factor = None
    if calibration_height_m and height > 0:
        calibration_factor = float(calibration_height_m / height)
        width *= calibration_factor
        depth *= calibration_factor
        height *= calibration_factor
        area *= calibration_factor ** 2

    return {
        "width": width,
        "depth": depth,
        "height": height,
        "area": area,
        "unit": "m",
        "calibration_factor": calibration_factor,
    }


@app.get("/healthz")
def healthcheck():
    return {"status": "ok"}


@app.post("/infer")
async def infer(
    request: Request,
    file: UploadFile = File(...),
    calibration_height_m: float | None = Form(None),
):
    _require_api_key(request)

    content_type = file.content_type or "application/octet-stream"
    if not content_type.startswith("image/"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="file must be an image")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="empty file")

    try:
        dimensions = _infer_dimensions(contents, calibration_height_m)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

    return {"dimensions": dimensions}
