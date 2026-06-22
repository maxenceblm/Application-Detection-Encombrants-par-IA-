#Détection en temps réel des objets depuis un flux vidéo webcam 
from ultralytics import YOLO
import cv2
model = YOLO("runs/detect/train-7/weights/best.pt")
cap = cv2.VideoCapture("london.mov")
while True:
    ret,frame = cap.read()
    if not ret :
        break 
    results = model(frame, device="mps",conf=0.25,imgsz=320)
    annotated = results[0].plot()
    cv2.imshow("Détections encombrants" , annotated)
    if cv2.waitKey(1) == ord("q") :  # Stop avec  q 
        break 
cap.release()
cv2.destroyAllWindows()