from pathlib import Path
from typing import Dict

import shutil
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from ultralytics import YOLO
import uvicorn

app = FastAPI(title="Simple Computer Vision ML Service")

WEIGHTS = {
    "nano": Path("weights/yolov8n.pt"),
    "small": Path("weights/yolov8s.pt"),
}

models: Dict[str, YOLO] = {}


@app.on_event("startup")
def load_models():
    for name, path in WEIGHTS.items():
        if not path.exists():
            raise RuntimeError(
                f"Missing {path}. Run: bash download_weights.sh"
            )
        models[name] = YOLO(str(path))


@app.get("/")
def root():
    return {
        "service": "YOLOv8 object detection API",
        "models": list(WEIGHTS.keys()),
        "usage": "POST /predict/{model_name} with form field: file",
    }


@app.post("/predict/{model_name}")
async def predict(model_name: str, file: UploadFile = File(...), conf: float = 0.25):
    if model_name not in models:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown model '{model_name}'. Use one of: {list(models.keys())}",
        )

    suffix = Path(file.filename or "image.jpg").suffix or ".jpg"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        image_path = tmp.name

    try:
        results = models[model_name](image_path, conf=conf, verbose=False)
        result = results[0]

        detections = []
        for box in result.boxes:
            detections.append(
                {
                    "class_id": int(box.cls[0]),
                    "class_name": result.names[int(box.cls[0])],
                    "confidence": float(box.conf[0]),
                    "bbox_xyxy": [float(x) for x in box.xyxy[0]],
                }
            )

        return {
            "model": model_name,
            "filename": file.filename,
            "detections": detections,
        }

    finally:
        Path(image_path).unlink(missing_ok=True)


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
