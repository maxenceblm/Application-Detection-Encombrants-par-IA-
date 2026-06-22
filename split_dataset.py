import os
import shutil
import random

path = "/Users/maxencebalme/Documents/Projet Encombrants/Application-Detection-Encombrants-par-IA-/dataset/appli"
train_images = f"{path}/train/images"
train_labels = f"{path}/train/labels"
valid_images = f"{path}/valid/images"
valid_labels = f"{path}/valid/labels"

os.makedirs(valid_images, exist_ok=True)
os.makedirs(valid_labels, exist_ok=True)

images = os.listdir(train_images)
random.shuffle(images)
val_images = images[:int(len(images) * 0.2)]

for img in val_images:
    shutil.move(f"{train_images}/{img}", f"{valid_images}/{img}")
    label = img.rsplit(".", 1)[0] + ".txt"
    if os.path.exists(f"{train_labels}/{label}"):
        shutil.move(f"{train_labels}/{label}", f"{valid_labels}/{label}")

print(f"Validation : {len(val_images)} images")
print(f"Train : {len(images) - len(val_images)} images")
