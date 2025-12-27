from __future__ import annotations

import os

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, status

from inference import infer_dimensions, warm_model

app = FastAPI(title="MoGe Dimensions API")


def _require_api_key(request: Request) -> None:
    expected = os.getenv("MOGE_API_KEY")
    if not expected:
        return
    auth = request.headers.get("authorization", "")
    if auth != f"Bearer {expected}":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid api key")




@app.get("/healthz")
def healthcheck():
    warm_model()
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
        payload = infer_dimensions(contents, calibration_height_m)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

    return payload
