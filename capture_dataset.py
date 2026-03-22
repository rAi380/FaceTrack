import cv2
import os

name = input("Enter student name: ")

dataset_path = "dataset/" + name

os.makedirs(dataset_path, exist_ok=True)

cap = cv2.VideoCapture(0)

count = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    cv2.imshow("Capture Faces", frame)

    key = cv2.waitKey(1)

    if key == ord('s'):
        img_path = dataset_path + "/" + str(count) + ".jpg"
        cv2.imwrite(img_path, frame)
        print("Saved:", img_path)
        count += 1

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()