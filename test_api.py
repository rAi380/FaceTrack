import requests
import os
print(os.getcwd())

url = "http://127.0.0.1:5000/recognize"

files = {
    "image": open(r"D:\☢️Mini Project☢️\test.jpg", "rb")
}

response = requests.post(url, files=files)

print(response.json())