from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pytesseract
import cv2
import numpy as np
import base64

app = FastAPI()

# 🚨 Allow CORS from all origins (or restrict to your GitHub Pages URL)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://nkmmns.github.io"],  # 👈 You can replace '*' with ["https://nkmmns.github.io"] for more security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ocr_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789'

@app.post("/ocr/")
async def extract_weight(file: UploadFile = File(...)):
    contents = await file.read()
    np_img = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    text = pytesseract.image_to_string(thresh, config=ocr_config).strip()

    # Annotate image
    cv2.putText(img, f"Weight: {text}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    _, buffer = cv2.imencode('.jpg', img)
    img_base64 = base64.b64encode(buffer).decode("utf-8")

    return {"weight": text, "image_base64": img_base64}
