import cv2
import requests
from ultralytics import YOLO
import time

# --- AYARLAR ---
API_URL = "http://127.0.0.1:8000/plaka"
last_process_time = 0
process_interval = 4  # 4 saniyede bir plaka sorgusu (İşlemciyi dinlendirir)
frame_skip = 10       # Her 10 karede bir analiz yap (Donmayı engeller)
frame_count = 0

# 1. Modeli yükle (Hız için 'n' nano model)
model = YOLO('yolov8n.pt') 

# 2. Kamera Ayarları (Iriun)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # Çözünürlüğü iyice düşürdük
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_FPS, 15)           # FPS'i sınırladık

print("🚗 Ultra Hafif İzleme Başladı...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame_count += 1
    
    # Ekranda görüntünün akması için analizi sadece belirli aralıklarla yapıyoruz
    if frame_count % frame_skip == 0:
        # imgsz=160 hızı aşırı artırır, 2011 Mac için hayat kurtarır
        results = model(frame, classes=[2, 3, 5, 7], conf=0.5, imgsz=160, verbose=False)
        
        for result in results:
            if len(result.boxes) > 0:
                cv2.putText(frame, "ARAC VAR", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                current_time = time.time()
                if current_time - last_process_time > process_interval:
                    _, img_encoded = cv2.imencode('.jpg', frame)
                    files = {'file': ('image.jpg', img_encoded.tobytes(), 'image/jpeg')}
                    try:
                        # API'ye gönderirken programın donmaması için bekleme yapmıyoruz
                        requests.post(API_URL, files=files, timeout=0.5) 
                        last_process_time = current_time
                    except:
                        pass

    # Görüntüyü göster
    cv2.imshow('Otopark Live', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()