import cv2
import requests

url = "http://127.0.0.1:5000/recognize"

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    cv2.imshow("CCTV Simulation", frame)

    # send frame every few seconds
    _, img_encoded = cv2.imencode('.jpg', frame)

    response = requests.post(
        url,
        files={"image": img_encoded.tobytes()}
    )

    print(response.json())

    if cv2.waitKey(3000) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()