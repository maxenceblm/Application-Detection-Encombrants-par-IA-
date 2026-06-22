from ultralytics import YOLO
import cv2
model =  YOLO("runs/detect/train-19/weights/best.pt")
frame = "Encombrants-trottoir1-1024x683.png"
results = model.predict(frame, conf=0.5, show=True)