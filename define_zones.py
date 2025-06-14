import cv2
import numpy as np
import json


camera_url = "rtsp://username:password@ip:port" #RTSP URL
cap = cv2.VideoCapture(camera_url)
ret, frame = cap.read()
cap.release()

if not ret:
    print("❌ Could not get frame")
    exit()

zone_points = []
drawing = True

def click_event(event, x, y, flags, param):
    global drawing
    if event == cv2.EVENT_LBUTTONDOWN and drawing:
        zone_points.append((x, y))

cv2.namedWindow("Define Zone")
cv2.setMouseCallback("Define Zone", click_event)

while True:
    temp = frame.copy()
    if len(zone_points) > 1:
        cv2.polylines(temp, [np.array(zone_points)], isClosed=True, color=(0,255,0), thickness=2)

    for pt in zone_points:
        cv2.circle(temp, pt, 5, (0,0,255), -1)

    cv2.imshow("Define Zone", temp)
    key = cv2.waitKey(1)

    if key == ord("s"):
        with open("zone.json", "w") as f:
            json.dump(zone_points, f)
        print("✅ Zone saved to zone.json")
        break
    elif key == ord("q"):
        break

cv2.destroyAllWindows()
