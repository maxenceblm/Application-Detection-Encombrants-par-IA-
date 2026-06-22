import cv2
import time
import numpy as np
from ultralytics import YOLO
from database import init_db, enregistrer_en_bdd, marquer_enleve , zone_deja_presente, set_camera_actif

model = YOLO("runs/detect/train-19/weights/best.pt")

init_db() #Initialisation BDD 
ID_CAMERA = 2
set_camera_actif(ID_CAMERA, 1)
cap = cv2.VideoCapture(0)
TAILLE_MIN = 5000
PERSISTANCE_MIN = 300 #
ABSENCE_MIN_SECONDES = 300 #
NOMS_CLASSES = {
    "canape": "Canapé",
    "carton": "Carton",
    "chaise": "Chaise",
    "commode": "Commode",
    "frigo": "Réfrigérateur",
    "machine_a_laver": "Machine à laver",
    "matelas": "Matelas",
    "sac_poubelle": "Sac poubelle",
    "table": "Table",
    "velo": "Vélo",
    "television": "Télévision"
}

print("Apprentissage du fond en cours...")
time.sleep(3)
ret, fond = cap.read()
fond_gris = cv2.cvtColor(fond, cv2.COLOR_BGR2GRAY)
fond_gris = cv2.GaussianBlur(fond_gris, (21, 21), 0)

roi = cv2.selectROI("Sélectionne la zone", fond, False)
cv2.destroyWindow("Sélectionne la zone")
x_roi, y_roi, w_roi, h_roi = roi

fond_gris_roi = fond_gris[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi]
print("Fond appris, détection active")

compteur_frames = {}
zones_sauvegardees = set()
zones_absentes = {}
zones_cooldown = {}
COOLDOWN_SECONDES = 30

def classifier_objet(frame, x, y, w, h):
    # Essai 1 : Crop estimé serré
    crop = frame[y:y+h, x:x+w]
    result = model(crop, conf=0.3, verbose=False)[0]
    if len(result.boxes) and float(result.boxes.conf.max()) >= 0.8:
        return result, crop
    # Essai 2 : crop élargi
    marge2 = 50
    x2 = max(0, x - marge2)
    y2 = max(0, y - marge2)
    x2w = min(frame.shape[1], x + w + marge2)# Min Hauteur Max et l'élargissement
    y2h = min(frame.shape[0], y + h + marge2)# Min Largeur Max et l'élargissement 
    crop_large = frame[y2:y2h, x2:x2w]
    result = model(crop_large, conf=0.8, verbose=False)[0]
    return result, crop_large

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gris = cv2.GaussianBlur(gris, (21, 21), 0)
        gris_roi = gris[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi]

        diff = cv2.absdiff(fond_gris_roi, gris_roi)
        _, mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        zones_actuelles = []
    
        for contour in contours:
            if cv2.contourArea(contour) > TAILLE_MIN:
                marge = 50
                x, y, w, h = cv2.boundingRect(contour)
                x_reel = x_roi + x
                y_reel = y_roi + y
                x_reel = max(0, x_reel - marge)
                y_reel = max(0, y_reel - marge)
                w = min(frame.shape[1] - x_reel, w + 2*marge)
                h = min(frame.shape[0] - y_reel, h + 2*marge)

                cx, cy = x_reel + w // 2, y_reel + h // 2
                zone_id = f"{cx//50}_{cy//50}"
                zones_actuelles.append(zone_id)
                compteur_frames[zone_id] = compteur_frames.get(zone_id, 0) + 1

                if compteur_frames[zone_id] >= PERSISTANCE_MIN:
                    cv2.rectangle(frame, (x_reel, y_reel), (x_reel+w, y_reel+h), (0, 255, 0), 2)
                    cv2.putText(frame, "Encombrant potentiel", (x_reel, max(10, y_reel-10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                en_cooldown = zone_id in zones_cooldown and time.time() - zones_cooldown[zone_id] < COOLDOWN_SECONDES
                if compteur_frames[zone_id] >= PERSISTANCE_MIN and zone_id not in zones_sauvegardees and not en_cooldown:
                    result, crop = classifier_objet(frame, x_reel, y_reel, w, h)
                    if len(result.boxes) and not zone_deja_presente(zone_id):
                        classe = model.names[int(result.boxes.cls[0])]
                        confiance = float(result.boxes.conf[0])
                        timeimg = time.strftime("%Y%m%d_%H%M%S", time.localtime())
                        photo_path = f"captures/{timeimg}.jpg"
                        cv2.imwrite(f"siteweb/static/captures/{timeimg}.jpg", crop)
                        zones_sauvegardees.add(zone_id)
                        zones_cooldown[zone_id] = time.time()
                        enregistrer_en_bdd(NOMS_CLASSES.get(classe,classe), confiance, photo_path, zone_id, ID_CAMERA)
                        print(f"Encombrant détecté : {classe} ({confiance:.2f})")
                        cv2.putText(frame, f"{classe} {confiance:.2f}", (x_reel, max(30, y_reel-30)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
        for zone_id in list(compteur_frames.keys()):
            if zone_id not in zones_actuelles:
                del compteur_frames[zone_id]
                if zone_id not in zones_absentes:
                    zones_absentes[zone_id] = time.time()
                elif (time.time() - zones_absentes[zone_id] >= ABSENCE_MIN_SECONDES):
                    marquer_enleve(zone_id)
                    zones_sauvegardees.discard(zone_id)
                    zones_absentes.pop(zone_id)
                    if zone_id in zones_cooldown :
                        del zones_cooldown[zone_id]
            else :
                zones_absentes.pop(zone_id,None) 

        cv2.imshow("Détection", frame)
        cv2.imshow("Masque", mask)

        if cv2.waitKey(1) == ord("q"):
            break
finally:
    set_camera_actif(ID_CAMERA, 0)
    cap.release()
    cv2.destroyAllWindows()
