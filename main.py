from pathlib import Path
import argparse

import shutil
import tempfile

from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
import uvicorn


parser = argparse.ArgumentParser(description="ML service")
parser.add_argument('-w', '--weights')
args = parser.parse_args()
print('args', args.weights)

app = FastAPI(title="Simple Computer Vision ML Service")
model: YOLO = None


@app.on_event('startup')
def load_model():
    global model
    model = YOLO(args.weights)
    print(model)


@app.get("/")
def root():
    return {
        "service": "YOLOv8 object detection API",
        "usage": "POST /predict with form field: file",
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...), conf: float = 0.25):
    suffix = Path(file.filename or "image.jpg").suffix or ".jpg"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        image_path = tmp.name

    try:
        results = model(image_path, conf=conf, verbose=False)
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
