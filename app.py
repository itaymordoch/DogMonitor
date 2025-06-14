import cv2
import numpy as np
import json
import time
import os
from datetime import datetime
from ultralytics import YOLO
import subprocess
import sys
import discord
import asyncio

os.makedirs("faces", exist_ok=True)
os.makedirs("alerts", exist_ok=True)

DTOKEN = '' # Discord Bot Token
DUSER_ID =   # Discord User ID

class MyClient(discord.Client):
    def __init__(self, msg, image_path=None, **kwargs):
        super().__init__(**kwargs)
        self.msg = msg
        self.image_path = image_path

    async def on_ready(self):
        print(f'Logged in as {self.user}')
        user = await self.fetch_user(DUSER_ID)
        await user.send(self.msg)
        if self.image_path:
            with open(self.image_path, 'rb') as f:
                await user.send(file=discord.File(f))
        await self.close()

def run_preset():
    subprocess.Popen([sys.executable, "preset.py"])

model = YOLO("yolov8n.pt")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

with open("zone.json", "r") as f:
    zone = np.array(json.load(f), dtype=np.int32)

camera_url = "rtsp://username:password@ip:port" #RTSP URL
cap = cv2.VideoCapture(camera_url)

if not cap.isOpened():
    print("❌ Could not connect to camera")
    exit()

run_preset()
start_time = time.time()
duration = 30 * 60
last_preset_time = start_time
last_human_alert = 0
last_dog_alert = 0
alert_interval = 5 * 60
last_yolo_time = 0

while time.time() - start_time < duration:
    ret, frame = cap.read()
    if not ret:
        time.sleep(1)
        continue

    now = time.time()
    if now - last_yolo_time >= 1:
        results = model.predict(frame, imgsz=640, conf=0.4, verbose=False)[0]
        last_yolo_time = now

        # Draw zone
        cv2.polylines(frame, [zone], isClosed=True, color=(0, 255, 0), thickness=2)

        detected_person = False

        for box in results.boxes:
            cls = int(box.cls[0])
            name = model.names[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            center = ((x1 + x2) // 2, (y1 + y2) // 2)

            if name == "person" and (now - last_human_alert >= alert_interval):
                print("🚨 HUMAN DETECTED 🚨")
                
                time.sleep(3)
                detected_person = True

            if name == "dog":
                inside = cv2.pointPolygonTest(zone, center, False)
                zone_top = min(pt[1] for pt in zone)
                if (inside >= 0 or center[1] <= zone_top) and (now - last_dog_alert >= alert_interval):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    dog_path = f"alerts/dog_{timestamp}.jpg"
                    cv2.imwrite(dog_path, frame)
                    print("🚨 KIRA IS ON THE COUCH 🚨")
                    client = MyClient("🚨 KIRA IS ON THE FUCKING COUCH 🚨", dog_path, intents=discord.Intents.default())
                    asyncio.run(client.start(DTOKEN))
                    last_dog_alert = now

        if detected_person and (now - last_human_alert >= alert_interval):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            if len(faces) > 0:
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                face_path = f"faces/human_{timestamp}.jpg"
                cv2.imwrite(face_path, frame)
                print("🚨 HUMAN DETECTED 🚨")
                client = MyClient("🚨 HUMAN DETECTED IN THE HOUSE 🚨", face_path, intents=discord.Intents.default())
                asyncio.run(client.start(DTOKEN))
                last_human_alert = now

    cv2.imshow("Monitoring", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    if time.time() - last_preset_time >= 30 * 60:
        run_preset()
        last_preset_time = time.time()

cap.release()
cv2.destroyAllWindows()
