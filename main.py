from fastapi import FastAPI, UploadFile, File
from img_ocr import run_ocr

app = FastAPI()

@app.get("/")
def root():
    return {"status": "OCR API running"}

@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    image_bytes = await file.read()
    text = run_ocr(image_bytes)
    return {"text": text}