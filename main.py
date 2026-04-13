from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import shutil
import os

from scripts.pipelines.video_to_evaluation import evaluation_pipeline

app = FastAPI()

UPLOAD_PATH = "uploads"
os.makedirs(UPLOAD_PATH, exist_ok=True)


# --------------------------------------------------
# FRONTEND PAGE
# --------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    # Read and return the HTML file directly — no Jinja2 needed
    with open("templates/index.html", "r") as f:
        content = f.read()
    return HTMLResponse(content=content)


# --------------------------------------------------
# VIDEO UPLOAD + PROCESS
# --------------------------------------------------
@app.post("/analyze")
async def analyze_video(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_PATH, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    issues = evaluation_pipeline(file_path)

    return JSONResponse({"issues": issues})