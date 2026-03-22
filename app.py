from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import os
import pickle
import re
import secrets
import sqlite3
import time
from mtcnn import MTCNN
from keras_facenet import FaceNet
from datetime import datetime
from werkzeug.security import check_password_hash, generate_password_hash

app = Flask(__name__)
CORS(app)

AUTH_DB = "attendance.db"
SESSION_DAYS = 7
_EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")


def init_auth_tables():
    conn = sqlite3.connect(AUTH_DB)
    c = conn.cursor()
    c.execute(
        """
    CREATE TABLE IF NOT EXISTS users (
        email TEXT PRIMARY KEY,
        password_hash TEXT NOT NULL,
        name TEXT NOT NULL,
        role TEXT NOT NULL CHECK(role IN ('Admin','Teacher'))
    )
    """
    )
    c.execute(
        """
    CREATE TABLE IF NOT EXISTS sessions (
        token TEXT PRIMARY KEY,
        email TEXT NOT NULL,
        expires_unix INTEGER NOT NULL
    )
    """
    )
    conn.commit()
    conn.close()


init_auth_tables()


def _normalize_email(email: str) -> str:
    return (email or "").strip().lower()


def _user_row_by_email(cursor, email: str):
    cursor.execute(
        "SELECT email, password_hash, name, role FROM users WHERE email = ?",
        (email,),
    )
    return cursor.fetchone()


def _session_user_from_token(token: str):
    if not token:
        return None
    now = int(time.time())
    conn = sqlite3.connect(AUTH_DB)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT email, expires_unix FROM sessions WHERE token = ?", (token,)
    )
    row = cursor.fetchone()
    if not row:
        conn.close()
        return None
    email, exp = row[0], row[1]
    if exp < now:
        cursor.execute("DELETE FROM sessions WHERE token = ?", (token,))
        conn.commit()
        conn.close()
        return None
    cursor.execute(
        "SELECT name, email, role FROM users WHERE email = ?", (email,)
    )
    u = cursor.fetchone()
    conn.close()
    if not u:
        return None
    return {"name": u[0], "email": u[1], "role": u[2]}


detector = MTCNN()
embedder = FaceNet()

EMBEDDINGS_PATH = "embeddings.pkl"
DATASET_ROOT = "dataset"
# Max L2 distance between stored embedding vectors and live face vector.
MATCH_THRESHOLD = 1.12


def _load_embeddings_store():
    """Load embeddings.pkl or start empty so Flask always starts."""
    if not os.path.isfile(EMBEDDINGS_PATH):
        return {"embeddings": [], "names": []}
    try:
        with open(EMBEDDINGS_PATH, "rb") as f:
            data = pickle.load(f)
        if not isinstance(data, dict) or "embeddings" not in data or "names" not in data:
            return {"embeddings": [], "names": []}
        return data
    except Exception:
        return {"embeddings": [], "names": []}


def _as_embedding_matrix(embeddings):
    """Normalize stored embeddings to 2-D numpy array (n, dim)."""
    arr = np.asarray(embeddings, dtype=np.float32)
    if arr.size == 0:
        # Placeholder shape until first registration supplies embedding width (FaceNet ~512).
        return np.empty((0, 512), dtype=np.float32)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    return arr


_data = _load_embeddings_store()
known_embeddings = _as_embedding_matrix(_data["embeddings"])
known_names = list(_data["names"])


def _detect_and_embed_from_bgr(image_bgr):
    """
    Produce one FaceNet embedding vector (1-D float32) from an OpenCV BGR frame.
    Keeps MTCNN + crop + resize in BGR so vectors stay consistent with a typical
    cv2.imdecode / imread → MTCNN → FaceNet pipeline and your stored embeddings.pkl.
    """
    if image_bgr is None:
        return None, "invalid_image"

    faces = detector.detect_faces(image_bgr)
    if len(faces) == 0:
        return None, "no_face"

    x, y, w, h = faces[0]["box"]
    h_img, w_img = image_bgr.shape[:2]
    x, y = max(0, int(x)), max(0, int(y))
    w, h = int(w), int(h)
    w = min(w, w_img - x)
    h = min(h, h_img - y)
    if w <= 1 or h <= 1:
        return None, "no_face"

    face = image_bgr[y : y + h, x : x + w]
    try:
        face = cv2.resize(face, (160, 160))
    except cv2.error:
        return None, "no_face"

    face = np.expand_dims(face, axis=0)
    embedding = embedder.embeddings(face)[0]
    return embedding, None


def extract_face_embedding(image_bgr):
    """
    Returns (embedding, None) on success, or (None, error_code) on failure.
    error_code: 'invalid_image' | 'no_face'
    """
    emb, err = _detect_and_embed_from_bgr(image_bgr)
    if err == "invalid_image":
        return None, "invalid_image"
    if err == "no_face":
        return None, "no_face"
    return emb, None


def dataset_dir_for_name(student_name):
    """Safe folder name under dataset/; mirrors a readable student name."""
    s = re.sub(r"[^\w\s\-]", "_", student_name.strip(), flags=re.UNICODE)
    s = "_".join(s.split())
    return (s or "student")[:120]


def persist_embeddings_file(embeddings_matrix, names_list):
    with open(EMBEDDINGS_PATH, "wb") as f:
        pickle.dump({"embeddings": embeddings_matrix, "names": names_list}, f)


def recognize_face(image_bgr):
    embedding, err = _detect_and_embed_from_bgr(image_bgr)
    if err == "invalid_image":
        return "No Face Found"
    if err == "no_face":
        return "No Face Found"

    matrix = _as_embedding_matrix(known_embeddings)
    if matrix.shape[0] == 0 or len(known_names) == 0:
        return "Unknown"

    min_dist = float("inf")
    name = "Unknown"
    n = min(matrix.shape[0], len(known_names))
    for i in range(n):
        dist = float(np.linalg.norm(matrix[i] - embedding))
        if dist < min_dist:
            min_dist = dist
            name = known_names[i]

    if min_dist < MATCH_THRESHOLD:
        return name

    return "Unknown"


def mark_attendance(name):

    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS attendance(
        name TEXT,
        date TEXT,
        time TEXT
    )
    """)

    now = datetime.now()

    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    cursor.execute(
        "INSERT INTO attendance VALUES (?,?,?)",
        (name, date, time)
    )

    conn.commit()
    conn.close()


@app.route("/")
def home():
    return "Face Attendance API Running"


@app.route("/students", methods=["GET"])
def list_students():
    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS students(
        name TEXT,
        roll_no TEXT PRIMARY KEY,
        course TEXT,
        year TEXT,
        section TEXT
    )
    """)
    cursor.execute(
        "SELECT name, roll_no, course, year, section FROM students ORDER BY rowid DESC"
    )
    rows = cursor.fetchall()
    conn.close()
    return jsonify(
        [
            {
                "name": r[0],
                "roll_no": r[1],
                "course": r[2],
                "year": r[3],
                "section": r[4],
            }
            for r in rows
        ]
    )


@app.route("/register", methods=["POST"])
def register():
    global known_embeddings, known_names

    name = (request.form.get("name") or "").strip()
    roll_no = (request.form.get("roll_no") or "").strip()
    course = (request.form.get("course") or "").strip()
    year = (request.form.get("year") or "").strip()
    section = (request.form.get("section") or "").strip()

    if not all([name, roll_no, course, year, section]):
        return jsonify({"error": "Missing required fields (name, roll_no, course, year, section)."}), 400

    if "image" not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    file = request.files["image"]
    if not file or file.filename == "":
        return jsonify({"error": "No image file selected."}), 400

    img_bytes = file.read()
    if not img_bytes:
        return jsonify({"error": "Empty image file."}), 400

    npimg = np.frombuffer(img_bytes, np.uint8)
    image_bgr = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    if image_bgr is None:
        return jsonify({"error": "Invalid image. Could not decode file."}), 400

    embedding, err = extract_face_embedding(image_bgr)
    if err == "no_face":
        return jsonify({"error": "No face detected in the image."}), 400
    if err == "invalid_image":
        return jsonify({"error": "Invalid image."}), 400

    if name in known_names:
        return jsonify({"error": "A student with this name is already enrolled for recognition."}), 400

    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS students(
        name TEXT,
        roll_no TEXT PRIMARY KEY,
        course TEXT,
        year TEXT,
        section TEXT
    )
    """)
    cursor.execute("SELECT 1 FROM students WHERE roll_no = ?", (roll_no,))
    if cursor.fetchone() is not None:
        conn.close()
        return jsonify({"error": "Roll number already registered."}), 400

    try:
        cursor.execute(
            "INSERT INTO students (name, roll_no, course, year, section) VALUES (?,?,?,?,?)",
            (name, roll_no, course, year, section),
        )
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        return jsonify({"error": "Roll number already registered."}), 400
    conn.close()

    subdir = dataset_dir_for_name(name)
    out_dir = os.path.join(DATASET_ROOT, subdir)
    os.makedirs(out_dir, exist_ok=True)
    image_path = os.path.join(out_dir, "image.jpg")
    if not cv2.imwrite(image_path, image_bgr):
        conn = sqlite3.connect("attendance.db")
        conn.execute("DELETE FROM students WHERE roll_no = ?", (roll_no,))
        conn.commit()
        conn.close()
        return jsonify({"error": "Failed to save image file."}), 500

    row = np.expand_dims(np.asarray(embedding, dtype=np.float32), axis=0)
    if len(known_names) == 0:
        new_matrix = row
    else:
        matrix = _as_embedding_matrix(known_embeddings)
        new_matrix = np.vstack([matrix, row])
    new_names = list(known_names) + [name]

    try:
        persist_embeddings_file(new_matrix, new_names)
    except OSError:
        try:
            if os.path.isfile(image_path):
                os.remove(image_path)
        except OSError:
            pass
        conn = sqlite3.connect("attendance.db")
        conn.execute("DELETE FROM students WHERE roll_no = ?", (roll_no,))
        conn.commit()
        conn.close()
        return jsonify({"error": "Failed to save embeddings file."}), 500

    known_embeddings = new_matrix
    known_names = new_names

    return jsonify({"ok": True, "message": "Student registered successfully."})


@app.route("/attendance", methods=["GET"])
def get_attendance():
    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS attendance(
        name TEXT,
        date TEXT,
        time TEXT
    )
    """)
    cursor.execute("SELECT name, date, time FROM attendance ORDER BY rowid DESC")
    rows = cursor.fetchall()
    conn.close()
    return jsonify([{"name": r[0], "date": r[1], "time": r[2]} for r in rows])


@app.route("/recognize", methods=["POST"])
def recognize():
    try:
        if "image" not in request.files:
            return jsonify({"recognized_person": "No Face Found"})

        file = request.files["image"]
        if not file or file.filename == "":
            return jsonify({"recognized_person": "No Face Found"})

        img_bytes = file.read()
        if not img_bytes:
            return jsonify({"recognized_person": "No Face Found"})

        npimg = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"recognized_person": "No Face Found"})

        name = recognize_face(frame)

        if name != "Unknown" and name != "No Face Found":
            mark_attendance(name)

        return jsonify({"recognized_person": name})
    except Exception as exc:
        print("[recognize]", repr(exc))
        return jsonify({"recognized_person": "No Face Found"})


def _issue_session(email: str) -> str:
    token = secrets.token_urlsafe(32)
    exp = int(time.time()) + SESSION_DAYS * 24 * 3600
    conn = sqlite3.connect(AUTH_DB)
    conn.execute("INSERT INTO sessions (token, email, expires_unix) VALUES (?,?,?)", (token, email, exp))
    conn.commit()
    conn.close()
    return token


@app.route("/auth/register", methods=["POST"])
def auth_register():
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    email = _normalize_email(data.get("email") or "")
    password = data.get("password") or ""
    role = data.get("role") or "Teacher"

    if not name or len(name) > 200:
        return jsonify({"error": "Name is required."}), 400
    if not _EMAIL_RE.match(email):
        return jsonify({"error": "Valid email is required."}), 400
    if len(password) < 8:
        return jsonify({"error": "Password must be at least 8 characters."}), 400
    if role not in ("Admin", "Teacher"):
        return jsonify({"error": "Invalid role."}), 400

    pw_hash = generate_password_hash(password)
    conn = sqlite3.connect(AUTH_DB)
    cur = conn.cursor()
    try:
        cur.execute(
            "INSERT INTO users (email, password_hash, name, role) VALUES (?,?,?,?)",
            (email, pw_hash, name, role),
        )
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        return jsonify({"error": "An account with this email already exists."}), 409
    conn.close()

    token = _issue_session(email)
    return (
        jsonify(
            {
                "token": token,
                "user": {"name": name, "email": email, "role": role},
            }
        ),
        201,
    )


@app.route("/auth/login", methods=["POST"])
def auth_login():
    data = request.get_json(silent=True) or {}
    email = _normalize_email(data.get("email") or "")
    password = data.get("password") or ""

    if not email or not password:
        return jsonify({"error": "Email and password are required."}), 400

    conn = sqlite3.connect(AUTH_DB)
    cur = conn.cursor()
    row = _user_row_by_email(cur, email)
    conn.close()
    if not row or not check_password_hash(row[1], password):
        return jsonify({"error": "Invalid email or password."}), 401

    name, role = row[2], row[3]
    token = _issue_session(email)
    return jsonify(
        {
            "token": token,
            "user": {"name": name, "email": email, "role": role},
        }
    )


@app.route("/auth/me", methods=["GET"])
def auth_me():
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return jsonify({"error": "Unauthorized"}), 401
    token = auth[7:].strip()
    user = _session_user_from_token(token)
    if not user:
        return jsonify({"error": "Unauthorized"}), 401
    return jsonify({"user": user})


@app.route("/auth/logout", methods=["POST"])
def auth_logout():
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        token = auth[7:].strip()
        conn = sqlite3.connect(AUTH_DB)
        conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
        conn.commit()
        conn.close()
    return jsonify({"ok": True})


if __name__ == "__main__":
    # 0.0.0.0 so the API works when the React app is opened via your LAN IP (e.g. http://172.16.x.x:8080).
    print("Face Attendance API → http://127.0.0.1:5000 (and http://<your-LAN-IP>:5000 on this machine)")
    app.run(debug=True, host="0.0.0.0", port=5000, use_reloader=False)