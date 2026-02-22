from fastapi import FastAPI, UploadFile, File
import numpy as np
import cv2
import easyocr

app = FastAPI()

reader = easyocr.Reader(['ko','en'])

@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    data = await file.read()

    np_arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2)

    result = reader.readtext(gray, detail=0, paragraph=True)
    text = "\n".join(result)

    return {"text": text}