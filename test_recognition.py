import cv2
import numpy as np
import pickle
from keras_facenet import FaceNet
from mtcnn import MTCNN

detector = MTCNN()
embedder = FaceNet()

with open("embeddings.pkl","rb") as f:
    data = pickle.load(f)

known_embeddings = data["embeddings"]
known_names = data["names"]

test_image = r"C:\Users\risha\.cache\kagglehub\datasets\jessicali9530\lfw-dataset\versions\4\lfw-deepfunneled\lfw-deepfunneled\Aaron_Eckhart\Aaron_Eckhart_0001.jpg"

img = cv2.imread(test_image)

faces = detector.detect_faces(img)
print("Faces detected:", len(faces))

x,y,w,h = faces[0]['box']

face = img[y:y+h, x:x+w]
face = cv2.resize(face,(160,160))

face = np.expand_dims(face,axis=0)

embedding = embedder.embeddings(face)[0]

min_dist = 100
name = "Unknown"

for i, known_embedding in enumerate(known_embeddings):

    dist = np.linalg.norm(known_embedding - embedding)

    if dist < min_dist:
        min_dist = dist
        name = known_names[i]

print("Recognized person:", name)
print("Distance:", min_dist)