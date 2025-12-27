# MoGe GPU Service

This repo runs MoGe on a GPU instance and returns room dimensions.

## Environment

- `MOGE_API_KEY` (optional): If set, requests must include `Authorization: Bearer <key>`
- `MOGE_DEVICE`: `cuda` or `cpu` (default `cuda`)
- `MOGE_MODEL_ID`: defaults to `Ruicheng/moge-2-vitl-normal`

## Run

```bash
pip install -r moge_service/requirements.txt
python -m uvicorn moge_service.app:app --host 0.0.0.0 --port 8001
```

## RunPod Serverless

1) Build and push the serverless image:

```bash
docker build -f Dockerfile.serverless -t <your-registry>/stagingequation-moge:latest .
docker push <your-registry>/stagingequation-moge:latest
```

2) Create a RunPod Serverless endpoint using that image.

3) RunPod handler expects:

```json
{
  "input": {
    "image_base64": "<base64>",
    "calibration_height_m": 2.4
  }
}
```

For warmup:

```json
{ "input": { "warm": true } }
```

## Request

`POST /infer` with multipart form-data:

- `file` (image)
- `calibration_height_m` (optional number)

Response:

```json
{
  "dimensions": {
    "width": 3.2,
    "depth": 4.1,
    "height": 2.6,
    "area": 13.12,
    "unit": "m",
    "calibration_factor": 1.05
  },
  "ceiling_overlay_base64": "data:image/png;base64,...",
  "ceiling_corners": [[120, 80], [520, 90], [110, 420], [540, 430]]
}
```
