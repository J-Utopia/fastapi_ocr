import pytesseract
import cv2
import numpy as np

# Render 환경용
pytesseract.pytesseract.tesseract_cmd = r"/usr/bin/tesseract"
# 로컬 테스트시
# pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"


def run_ocr(image_bytes):

    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2)

    text = pytesseract.image_to_string(
        gray,
        lang="kor+eng",
        config="--psm 3"
    )

    return text