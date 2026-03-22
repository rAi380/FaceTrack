# 🎓 FaceTrack - Smart Face Attendance System

FaceTrack is an AI-powered smart attendance system that uses face recognition to automatically mark student attendance. It replaces traditional manual attendance with a fast, secure, and contactless solution.

---

## 🚀 Features

* 🎥 Face Recognition using AI (FaceNet)
* 📷 Real-time image capture (Camera integration)
* 🧠 Automatic attendance marking
* 🗄️ SQLite database for storing attendance
* 👨‍🎓 Student registration with photo upload
* 📊 Attendance dashboard (Frontend UI)
* ⚡ Fast and accurate recognition using embeddings

---

## 🧠 How It Works

1. Student registers with photo
2. Image is processed using FaceNet
3. Face is converted into a 128-dimensional embedding
4. Embeddings are stored in the system
5. During attendance:

   * Camera captures face
   * System compares embeddings
   * If matched → attendance is marked

---

## 🏗️ Project Architecture

Frontend (Dashboard UI)
↓
Flask Backend API
↓
Face Detection (MTCNN)
↓
FaceNet Model (Embeddings)
↓
Compare with Stored Data
↓
SQLite Database (Attendance)

---

## 🛠️ Tech Stack

### 🔹 Frontend

* HTML / CSS / JavaScript (Lovable AI UI)

### 🔹 Backend

* Python
* Flask

### 🔹 AI & Computer Vision

* FaceNet
* OpenCV
* MTCNN
* NumPy

### 🔹 Database

* SQLite (attendance.db)

---

## 📂 Project Structure

```
├── app.py
├── generate_embeddings.py
├── embeddings.pkl
├── attendance.db
├── dataset/
├── frontend/
└── README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/smart-attendance-hub.git
cd smart-attendance-hub
```

---

### 2️⃣ Install Backend Dependencies

```bash
pip install flask opencv-python numpy mtcnn keras-facenet flask-cors
```

---

### 3️⃣ Run Backend

```bash
python app.py
```

---

### 4️⃣ Run Frontend

```bash
npm install
npm run dev
```

---

## 📸 API Endpoints

### 🔹 Face Recognition

```
POST /recognize
```

### 🔹 Get Attendance

```
GET /attendance
```

### 🔹 Register Student

```
POST /register
```

---

## 🎯 Use Cases

* 🏫 College/School attendance system
* 🏢 Office employee attendance
* 🧑‍💼 Secure identity verification

---

## 💡 Future Enhancements

* ☁️ Firebase cloud database integration
* 🎥 Real-time CCTV integration
* 📊 Advanced analytics dashboard
* 🔒 Multi-face detection & tracking
* 📱 Mobile app integration

---

## 🧠 Project Highlights

* AI-based automation (Face Recognition)
* Real-time processing
* Scalable architecture (CCTV-ready)
* Contactless attendance system

---

## 👨‍💻 Author

Developed by: Rishabh Rai, Harsh Tiwari, Harsh Goyal, Animish Gupta

---

## ⭐ Support

If you like this project, give it a ⭐ on GitHub!
