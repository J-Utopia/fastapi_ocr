import pytesseract
import cv2
import numpy as np

import platform
import pytesseract

if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"
else:
    pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"


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