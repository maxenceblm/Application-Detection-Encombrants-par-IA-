import cv2
import time
import numpy as np  

cap = cv2.VideoCapture("london.mov")
TAILLE_MIN = 2000
PERSISTANCE_MIN = 300

fond = cv2.imread("london.png")
fond = cv2.resize(fond, (cap.get(cv2.CAP_PROP_FRAME_WIDTH).__int__(), cap.get(cv2.CAP_PROP_FRAME_HEIGHT).__int__()))
fond_gris = cv2.cvtColor(fond, cv2.COLOR_BGR2GRAY)
fond_gris = cv2.GaussianBlur(fond_gris, (21, 21), 0)

premiere_frame = cv2.imread("london.png")
roi = cv2.selectROI("Sélectionne la zone", premiere_frame, False)
cv2.destroyWindow("Sélectionne la zone")
x_roi, y_roi, w_roi, h_roi = roi # Coordonnées de la zone redimensionnée

# Appliquer la ROI sur le fond
fond_gris_roi = fond_gris[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi]
print("Fond appris, détection active")

compteur_frames = {}
zones_sauvegardees = set()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gris = cv2.GaussianBlur(gris, (21, 21), 0)
    gris_roi = gris[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi]

    diff = cv2.absdiff(fond_gris_roi, gris_roi)
    _, mask = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
    
    # Réduction du Bruit 
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
   
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #Détecte les contours 

    zones_actuelles = []

    for contour in contours:
        if cv2.contourArea(contour) > TAILLE_MIN: #  Aire Total de l'objet 
            marge = 50
            x, y, w, h = cv2.boundingRect(contour) #Rectangle autour de l'objet 
            # Remettre dans le repère de la frame complète
            x_reel = x_roi + x
            y_reel = y_roi + y
            x_reel = max(0, x_reel - marge) #Cas ou la marge ferait sortir du cadre 
            y_reel = max(0, y_reel - marge) #Idem 
            w = min(frame.shape[1] - x_reel, w + 2*marge)
            h = min(frame.shape[0] - y_reel, h + 2*marge)

            cx, cy = x_reel + w // 2, y_reel + h // 2
            zone_id = f"{cx//50}_{cy//50}"
            zones_actuelles.append(zone_id)
            compteur_frames[zone_id] = compteur_frames.get(zone_id, 0) + 1 #Incrémente le nombre de frames 

            if compteur_frames[zone_id] >= PERSISTANCE_MIN:
                cv2.rectangle(frame, (x_reel, y_reel), (x_reel+w, y_reel+h), (0, 255, 0), 2)
                cv2.putText(frame, "Encombrant potentiel", (x_reel, max(10, y_reel-10)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            if compteur_frames[zone_id] >= PERSISTANCE_MIN and zone_id not in zones_sauvegardees:
                crop = frame[y_reel:y_reel+h, x_reel:x_reel+w]
                timeimg = time.strftime("%Y%m%d_%H%M%S", time.localtime())
                zone_masque = mask[y:y+h, x:x+w]
                ratio_blanc = cv2.countNonZero(zone_masque) / (w * h)
                if ratio_blanc > 0.3: #30% de changements de pixels dans la zone
                    cv2.imwrite(f"captures/{timeimg}.jpg", crop)
                    zones_sauvegardees.add(zone_id)
    #Effacer les zones qui ne sont plus détectés depuis un moment
    for zone_id in list(compteur_frames.keys()):
        if zone_id not in zones_actuelles:
            del compteur_frames[zone_id]
            zones_sauvegardees.discard(zone_id)

    cv2.imshow("Détection", frame)
    cv2.imshow("Masque", mask)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()