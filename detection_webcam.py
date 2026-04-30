#Détection en temps réel des objets depuis un flux vidéo webcam 
from ultralytics import YOLO
import cv2
model = YOLO()
cap = cv2.VideoCapture(0)
while True:
    ret,frame = cap.read()
    if not ret :
        break 
    results = model(frame)
    annotated = results[0].plot()
    cv2.imshow("Détections encombrants" , annotated)
    if cv2.waitKey(1) == ord("q") :  # Stop avec  q 
        break 
cap.release()
cv2.destroyAllWindows()