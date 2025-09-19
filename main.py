from fastapi import FastAPI, UploadFile, File
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import shutil

app = FastAPI()

# ★CORS設定★
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # 必要に応じて限定可
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path(__file__).parent / "image"
UPLOAD_DIR.mkdir(exist_ok=True)

@app.post("/upload", response_class=PlainTextResponse)
async def upload_file(file: UploadFile = File(...)):
    file_path = UPLOAD_DIR / file.filename
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return "File uploaded: " + file.filename

@app.get("/")
def root():
    return {"status": "ok"}
