# Application détection d'encombrants par IA 
Ce projet consiste à développer une application permettant la détection automatique
d’encombrants à partir d’un flux vidéo issu d’une caméra. La première étape est de
détecter un objet anormal sur une scène fixe avec une persistance dans le temps pour
éviter de faux positifs. La seconde étape est celle de classification, elle consiste à
identifier le type d’objet (carton, meuble, électroménager …). Ces informations sont
ensuite enregistrées dans une base de données dynamique utilisée par une application
web permettant de visualiser, filtrer et suivre les signalements. Le projet peut inclure
par la suite une logique métier proposant par exemple un type de véhicule adapté pour
l’intervention. Les diﬃcultés seront notamment la détection d’encombrants dans un
environnement réel (variations de lumières et mouvements) ainsi que dans la gestion
temporelle des objets détectés.

# Etape 1 : Préparation et Architecture du Projet 
- Création du dépot GitHub 
- Import des bibliothèsques python et téléchargements à faire si besoin 
- Installation d'un environnement virtuel avec toutes les dépendances dessus
- Création du cahier des charges 

# Etape 2 : Détection des objets et logique de persistance temporelle 
- Apprentissage de l'image source de la caméra 
- Detection grâce aux niveaux de gris , d'un nouvel objet qui apparait dans le champ de l'image  
- Appliquer une logique de persistance temporelle de x secondes pour sauvegarder la zone de détection 
- Mettre en place des zones de détection pour indiquer quels zones sont utiles (ex : écarter le ciel )

# Etape 3 : Classification des Encombrants par annotation 