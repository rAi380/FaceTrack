import cv2
import numpy as np
from keras_facenet import FaceNet
from mtcnn import MTCNN

detector = MTCNN()
embedder = FaceNet()

# Use two images of SAME person
img1 = cv2.imread("dataset_small/Aaron_Eckhart/Aaron_Eckhart_0001.jpg")
img2 = cv2.imread("dataset_small/Aaron_Eckhart/Aaron_Eckhart_0002.jpg")

def get_embedding(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(img_rgb)

    x,y,w,h = faces[0]['box']
    x,y = max(0,x), max(0,y)

    face = img_rgb[y:y+h, x:x+w]
    face = cv2.resize(face, (160,160))
    face = np.expand_dims(face, axis=0)

    return embedder.embeddings(face)[0]

emb1 = get_embedding(img1)
emb2 = get_embedding(img2)

dist = np.linalg.norm(emb1 - emb2)

print("Same person distance:", dist)