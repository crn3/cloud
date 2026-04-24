from pathlib import Path
from datetime import datetime, timezone
import csv
import shutil

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request

APP_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = APP_DIR / "uploads"
LOG_CSV = APP_DIR / "upload_log.csv"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Satellite Scene Receiver")


def append_log(row: dict):
    file_exists = LOG_CSV.exists()
    fieldnames = [
        "timestamp_utc",
        "scene_id",
        "mode",
        "filename",
        "bytes_received",
        "client_host",
    ]

    with open(LOG_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload_scene")
async def upload_scene(
    request: Request,
    scene_id: str = Form(...),
    mode: str = Form(...),   # all | unet | unetplusplus
    file: UploadFile = File(...),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    if not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Only .zip uploads are accepted")

    scene_dir = UPLOAD_DIR / mode / scene_id
    scene_dir.mkdir(parents=True, exist_ok=True)

    save_path = scene_dir / file.filename

    with open(save_path, "wb") as out_f:
        shutil.copyfileobj(file.file, out_f)

    bytes_received = save_path.stat().st_size

    append_log(
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "scene_id": scene_id,
            "mode": mode,
            "filename": file.filename,
            "bytes_received": bytes_received,
            "client_host": request.client.host if request.client else "",
        }
    )

    return {
        "status": "ok",
        "scene_id": scene_id,
        "mode": mode,
        "filename": file.filename,
        "bytes_received": bytes_received,
    }