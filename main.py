from fastapi import FastAPI, UploadFile, File
from fastapi.responses import PlainTextResponse
from pathlib import Path
import os

app = FastAPI()

# 保存先ディレクトリ（Renderのコンテナ上）
BASE_DIR = Path(__file__).resolve().parent
IMAGE_DIR = BASE_DIR / "image"
IMAGE_DIR.mkdir(exist_ok=True)

@app.post("/upload", response_class=PlainTextResponse)
async def upload_file(file: UploadFile = File(...)):
    # 保存ファイルパス
    save_path = IMAGE_DIR / file.filename
    try:
        with open(save_path, "wb") as f:
            content = await file.read()
            f.write(content)
        return f"Saved to {save_path.name}"
    except Exception as e:
        return PlainTextResponse(str(e), status_code=500)

@app.get("/")
async def root():
    return {"status": "ok", "saved_dir": str(IMAGE_DIR)}
