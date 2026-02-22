from fastapi import FastAPI, UploadFile, File
import numpy as np
import cv2
import os

# 🔥 GPU 완전 차단 (CUDA 로딩 방지)
os.environ["CUDA_VISIBLE_DEVICES"] = ""

app = FastAPI()

reader = None  # 모델 전역 변수

@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    global reader

    # 🔥 요청 시점에만 모델 로딩
    if reader is None:
        import easyocr
        reader = easyocr.Reader(
            ['ko', 'en'],
            gpu=False,          # GPU 사용 안함
            verbose=False
        )

    # 파일 읽기
    data = await file.read()
    np_arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "invalid image"}

    # 전처리
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2)

    # OCR 실행
    result = reader.readtext(gray, detail=0, paragraph=True)

    text = "\n".join(result).strip()

    return {"text": text}