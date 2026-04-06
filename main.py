from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import cv2
import easyocr
import numpy as np
import re

app = FastAPI()

class PlakaResponse(BaseModel):
    plaka: str | None
    status: str

# 2011 Mac için CPU modunda ve Türkçe destekli başlatıyoruz
print("🚀 OCR yükleniyor (TR/EN)...")
reader = easyocr.Reader(['tr', 'en'], gpu=False) 

def preprocess_image(img):
    # Plaka okumayı kolaylaştırmak için gri ton ve kontrast artırma
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Gürültü azaltma
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    return gray

@app.post("/plaka", response_model=PlakaResponse)
async def plaka_oku(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return {"plaka": None, "status": "Görüntü okunamadı"}

    # Görüntüyü iyileştir
    processed_img = preprocess_image(img)

    # OCR İşlemi
    results = reader.readtext(processed_img)
    
    full_text = "".join([res[1] for res in results]).upper().replace(" ", "")
    # "TR" ibaresini ve gereksiz karakterleri temizle
    full_text = re.sub(r'[^A-Z0-0]', '', full_text.replace("TR", ""))

    print(f"🔎 Analiz Edilen Metin: {full_text}")

    # Gelişmiş TR Plaka Regex: (Şehir)(Harf)(Rakam)
    # Örn: 34ABC123, 06A1234, 34KLM56
    match = re.search(r'(\d{2})([A-Z]{1,3})(\d{2,4})', full_text)

    if match:
        city, letters, numbers = match.groups()
        # Ticari standartlara göre plaka formatı doğrula
        final_plate = f"{city}{letters}{numbers}"
        print(f"✅ Plaka Onaylandı: {final_plate}")
        return {"plaka": final_plate, "status": "Başarılı"}

    return {"plaka": None, "status": "Plaka bulunamadı"}