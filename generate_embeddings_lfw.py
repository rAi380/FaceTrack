import os
import cv2
import numpy as np
import pickle
from mtcnn import MTCNN
from keras_facenet import FaceNet

#dataset_path = r"C:\Users\risha\.cache\kagglehub\datasets\jessicali9530\lfw-dataset\versions\4\lfw-deepfunneled\lfw-deepfunneled"
dataset_path = "dataset_small"
detector = MTCNN()
embedder = FaceNet()

embeddings = []
names = []

for person_name in os.listdir(dataset_path):

    person_folder = os.path.join(dataset_path, person_name)

    if not os.path.isdir(person_folder):
        continue

    print("Processing:", person_name)

    for image_name in os.listdir(person_folder):

        image_path = os.path.join(person_folder, image_name)

        img = cv2.imread(image_path)

        if img is None:
            continue

        # convert BGR → RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        faces = detector.detect_faces(img_rgb)

        if len(faces) == 0:
            continue

        x, y, w, h = faces[0]["box"]

        x = max(0, x)
        y = max(0, y)

        face = img_rgb[y:y+h, x:x+w]

        try:
            face = cv2.resize(face, (160,160))
        except:
            continue

        face = np.expand_dims(face, axis=0)

        embedding = embedder.embeddings(face)[0]

        embeddings.append(embedding)
        names.append(person_name)

print("Total faces processed:", len(embeddings))

data = {
    "embeddings": embeddings,
    "names": names
}

with open("embeddings.pkl","wb") as f:
    pickle.dump(data,f)

print("Embeddings saved successfully")