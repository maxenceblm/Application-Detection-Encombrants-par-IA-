import cv2
import time
import numpy as np  

cap = cv2.VideoCapture(0)
TAILLE_MIN = 4000 #Taille Minimum pour considérer un objet 
PERSISTANCE_MIN = 90 #Temps minimum pour considérer un objet 
print("Apprentissage du fond en cours ... ")
time.sleep(3)
ret, fond = cap.read()
# fond = cv2.imread("beach_fond.png")
fond = cv2.resize(fond, (cap.get(cv2.CAP_PROP_FRAME_WIDTH).__int__(), cap.get(cv2.CAP_PROP_FRAME_HEIGHT).__int__()))
fond_gris = cv2.cvtColor(fond, cv2.COLOR_BGR2GRAY)
fond_gris = cv2.GaussianBlur(fond_gris, (21, 21), 0)
print("Fond appris, détection active")

compteur_frames = {}
zones_sauvegardees = set()

while True :

    ret , frame = cap.read()
    if not ret :
        break
    gris = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gris = cv2.GaussianBlur(gris,(21,21),0)

    diff = cv2.absdiff(fond_gris , gris)

    _,mask = cv2.threshold(diff,20,255,cv2.THRESH_BINARY)


    #Supprimer le bruit avec un filtre morphologique
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
    mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)

    #Estimer contours zone changement 
    contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    zones_actuelles = []

    for contour in contours : 
        print(f"Surface : {cv2.contourArea(contour)}")
        if cv2.contourArea(contour) > TAILLE_MIN:
            # On applique une marge pour éviter de crop l'objet pas entier 
            marge = 50  # pixels
            x,y,w,h = cv2.boundingRect(contour)
            x = max(0, x - marge)
            y = max(0, y - marge)
            w = min(frame.shape[1] - x, w + 2*marge)
            h = min(frame.shape[0] - y, h + 2*marge)
            cx, cy = x + w // 2, y + h // 2
            zone_id = f"{cx//50}_{cy//50}"
            zones_actuelles.append(zone_id)
            compteur_frames[zone_id] = compteur_frames.get(zone_id, 0) + 1

            if compteur_frames[zone_id] >= PERSISTANCE_MIN and zone_id not in zones_sauvegardees:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2) # Affichage rectangle vert autour
                cv2.putText(frame, "Encombrant potentiel", (x, y-10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                crop = frame[y:y+h, x:x+w] # Découpage Image détecté 
                timeimg = time.strftime("%Y%m%d_%H%M%S", time.localtime()) #récuparation temps de la frame
                zone_masque = mask[y:y+h, x:x+w]
                ratio_blanc = cv2.countNonZero(zone_masque) / (w * h)
                if ratio_blanc > 0.3:
                    cv2.imwrite(f"captures/{timeimg}.jpg", crop)
                    zones_sauvegardees.add(zone_id)

    # supprimer les zones qui ne sont plus détectées 
    for zone_id in list(compteur_frames.keys()):
        if zone_id not in zones_actuelles:
            del compteur_frames[zone_id]
            zones_sauvegardees.discard(zone_id)
    cv2.imshow("Détection",frame)
    cv2.imshow('Masque', mask )
    if cv2.waitKey(1) == ord("q") :
        break 

cap.release()
cv2.destroyAllWindows()