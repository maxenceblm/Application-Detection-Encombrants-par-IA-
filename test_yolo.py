#Fichier pour tester les modèles facilement sur une image 
from ultralytics import YOLO

model = YOLO("yolov8m.pt")  #charge le modèle 
results = model("https://www.grandparissud.fr/app/uploads/2023/03/dechets-encombrants-962x500.jpeg")    #image passé dans le modèle
results[0].show()
