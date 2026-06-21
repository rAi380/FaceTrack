from dotenv import load_dotenv
load_dotenv()

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
import traceback
import smtplib
from email.message import EmailMessage
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
    # Allow session tokens for both teachers (email) and students (roll_no).
    c.execute("PRAGMA table_info(sessions)")
    s_cols = [row[1] for row in c.fetchall()]
    if "roll_no" not in s_cols:
        c.execute("ALTER TABLE sessions ADD COLUMN roll_no TEXT")
    if "role" not in s_cols:
        c.execute("ALTER TABLE sessions ADD COLUMN role TEXT")
    if "name" not in s_cols:
        c.execute("ALTER TABLE sessions ADD COLUMN name TEXT")
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
        "SELECT email, roll_no, role, name, expires_unix FROM sessions WHERE token = ?",
        (token,),
    )
    row = cursor.fetchone()
    if not row:
        conn.close()
        return None
    email, roll_no, role, name, exp = row[0], row[1], row[2], row[3], row[4]
    if exp < now:
        cursor.execute("DELETE FROM sessions WHERE token = ?", (token,))
        conn.commit()
        conn.close()
        return None
    # If session row already has identity, return it (works for students and teachers).
    if role and name and (email or roll_no):
        conn.close()
        u = {"name": name, "role": role}
        if email:
            u["email"] = email
        if roll_no:
            u["roll_no"] = roll_no
        return u

    # Backward-compatible: old teacher sessions only stored email; resolve from users table.
    if not email:
        conn.close()
        return None
    cursor.execute("SELECT name, email, role FROM users WHERE email = ?", (email,))
    u = cursor.fetchone()
    conn.close()
    if not u:
        return None
    return {"name": u[0], "email": u[1], "role": u[2]}


detector = MTCNN()
embedder = FaceNet()

EMBEDDINGS_PATH = "embeddings.pkl"
# Max L2 distance between stored embedding vectors and live face vector.
MATCH_THRESHOLD = 1.12

PRESENT_STATUS = "Present"
EDITED_STATUS = "Edited"
VERIFIED_STATUS = "Verified"

SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASS = os.getenv("SMTP_PASS", "")
SMTP_FROM = os.getenv("SMTP_FROM", "")

EMAIL_SUBJECT = "Attendance Verification - FaceTrack"


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
    try:
        embedding = embedder.embeddings(face)[0]
    except Exception:
        return None, "no_face"
    return embedding, None


def extract_face_embedding(image_bgr):
    """
    Returns (embedding, None) on success, or (None, error_code) on failure.
    error_code: 'invalid_image' | 'no_face' | 'encode_error'
    """
    try:
        emb, err = _detect_and_embed_from_bgr(image_bgr)
    except Exception:
        return None, "encode_error"
    if err == "invalid_image":
        return None, "invalid_image"
    if err == "no_face":
        return None, "no_face"
    return emb, None


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


def _detect_and_embed_multiple_from_bgr(image_bgr, max_faces: int = 10):
    """
    Detect all faces in the frame and return FaceNet embeddings for each.
    Returns (embeddings, error_code). error_code is 'invalid_image' | 'no_face'.
    """
    if image_bgr is None:
        return [], "invalid_image"

    faces = detector.detect_faces(image_bgr)
    if not faces:
        return [], "no_face"

    embeddings = []
    h_img, w_img = image_bgr.shape[:2]

    for face in faces[:max_faces]:
        box = face.get("box")
        if not box or len(box) != 4:
            continue
        x, y, w, h = box

        x, y, w, h = int(x), int(y), int(w), int(h)
        x = max(0, x)
        y = max(0, y)
        w = max(1, min(w, w_img - x))
        h = max(1, min(h, h_img - y))

        if w <= 1 or h <= 1:
            continue

        crop = image_bgr[y : y + h, x : x + w]
        try:
            crop = cv2.resize(crop, (160, 160))
        except cv2.error:
            continue

        crop = np.expand_dims(crop, axis=0)
        embedding = embedder.embeddings(crop)[0]
        embeddings.append(embedding)

    if not embeddings:
        return [], "no_face"
    return embeddings, None


def recognize_faces(image_bgr):
    """
    Recognize multiple faces in one frame.
    Returns a list of unique recognized names (no duplicates).
    """
    embeddings, err = _detect_and_embed_multiple_from_bgr(image_bgr)
    if err:
        return []

    if len(known_names) == 0 or known_embeddings.shape[0] == 0:
        return []

    recognized = []
    seen = set()
    for emb in embeddings:
        dists = np.linalg.norm(known_embeddings - emb, axis=1)
        i = int(np.argmin(dists))
        if float(dists[i]) < MATCH_THRESHOLD:
            name = known_names[i]
            if name not in seen:
                seen.add(name)
                recognized.append(name)
    return recognized


def recognize_faces_with_boxes(image_bgr, max_faces: int = 10):
    """
    Like recognize_faces(), but returns per-face results including bounding boxes.
    Only returns registered (matched) faces.
    Format: [{ "name": str, "box": [x,y,w,h] }, ...]
    """
    if image_bgr is None:
        return []
    faces = detector.detect_faces(image_bgr)
    if not faces:
        return []
    if len(known_names) == 0 or known_embeddings.shape[0] == 0:
        return []

    out = []
    seen = set()
    h_img, w_img = image_bgr.shape[:2]

    for face in faces[:max_faces]:
        box = face.get("box")
        if not box or len(box) != 4:
            continue
        x, y, w, h = [int(v) for v in box]
        x = max(0, x)
        y = max(0, y)
        w = max(1, min(w, w_img - x))
        h = max(1, min(h, h_img - y))
        if w <= 1 or h <= 1:
            continue
        crop = image_bgr[y : y + h, x : x + w]
        try:
            crop = cv2.resize(crop, (160, 160))
        except cv2.error:
            continue
        crop = np.expand_dims(crop, axis=0)
        emb = embedder.embeddings(crop)[0]
        dists = np.linalg.norm(known_embeddings - emb, axis=1)
        i = int(np.argmin(dists))
        if float(dists[i]) < MATCH_THRESHOLD:
            name = known_names[i]
            if name in seen:
                continue
            seen.add(name)
            out.append({"name": name, "box": [x, y, w, h]})
    return out


def mark_attendance(name):
    return mark_attendance_with_status(name=name, status=PRESENT_STATUS)


def ensure_attendance_schema():
    """
    Ensure the `attendance` table has the `status` column.
    Older versions created `attendance` without status, so we run a light migration.
    """
    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS attendance(
            name TEXT,
            date TEXT,
            time TEXT
        )
        """
    )
    cursor.execute("PRAGMA table_info(attendance)")
    cols = [row[1] for row in cursor.fetchall()]
    if "status" not in cols:
        cursor.execute("ALTER TABLE attendance ADD COLUMN status TEXT")
    conn.commit()
    conn.close()


def ensure_students_schema():
    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS students(
            name TEXT,
            roll_no TEXT PRIMARY KEY,
            course TEXT,
            year TEXT,
            section TEXT
        )
        """
    )
    cursor.execute("PRAGMA table_info(students)")
    cols = [row[1] for row in cursor.fetchall()]
    if "role" not in cols:
        cursor.execute("ALTER TABLE students ADD COLUMN role TEXT")
        cursor.execute("UPDATE students SET role = 'student' WHERE role IS NULL")
    if "password_hash" not in cols:
        cursor.execute("ALTER TABLE students ADD COLUMN password_hash TEXT")
    conn.commit()
    conn.close()


def mark_attendance_with_status(name: str, status: str = PRESENT_STATUS) -> bool:
    """
    Mark attendance once per student per day.
    Returns True if a row was inserted, False if it already existed.
    """
    ensure_attendance_schema()

    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()

    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    cursor.execute(
        "SELECT 1 FROM attendance WHERE name = ? AND date = ?",
        (name, date),
    )
    if cursor.fetchone() is not None:
        conn.close()
        return False

    cursor.execute(
        "INSERT INTO attendance (name, date, time, status) VALUES (?,?,?,?)",
        (name, date, time, status),
    )
    conn.commit()
    conn.close()
    return True


def _require_auth_user():
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return None, (jsonify({"error": "Unauthorized"}), 401)
    token = auth[7:].strip()
    user = _session_user_from_token(token)
    if not user:
        return None, (jsonify({"error": "Unauthorized"}), 401)
    return user, None


def _send_attendance_email(to_email: str, students: list[str], verify_url: str, date: str, time_label: str):
    if not SMTP_USER or not SMTP_PASS or not SMTP_FROM:
        raise RuntimeError("Email is not configured on the server. Set SMTP_USER/SMTP_PASS/SMTP_FROM env vars.")

    student_lines = "\n".join([f"- {s}" for s in students])
    msg = EmailMessage()
    msg["From"] = SMTP_FROM or SMTP_USER
    msg["To"] = to_email
    msg["Subject"] = EMAIL_SUBJECT
    msg.set_content(
        f"Students Present:\n{student_lines if student_lines else '- (none)'}\n\n"
        f"Date: {date}\n"
        f"Time: {time_label}\n\n"
        f"Verify Attendance:\n{verify_url}\n"
    )

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as smtp:
        smtp.starttls()
        smtp.login(SMTP_USER, SMTP_PASS)
        smtp.send_message(msg)


@app.route("/")
def home():
    return "Face Attendance API Running"


@app.route("/students", methods=["GET"])
def list_students():
    ensure_students_schema()
    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()
    cursor.execute(
        "SELECT name, roll_no, course, year, section, role FROM students ORDER BY rowid DESC"
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
                "role": r[5] if r[5] else "student",
            }
            for r in rows
        ]
    )


@app.route("/register", methods=["POST"])
def register():
    global known_embeddings, known_names

    try:
        return _register_student_post()
    except Exception:
        traceback.print_exc()
        return jsonify(
            {
                "error": "Registration failed on the server. See the Flask terminal for the full error.",
            }
        ), 500


def _register_student_post():
    global known_embeddings, known_names

    # Public student self-registration (no login). Teacher uses dashboard only to view students.
    role = (request.form.get("role") or "").strip().lower()
    if role != "student":
        return jsonify({"error": "Only student self-registration is allowed here."}), 403

    name = (request.form.get("name") or "").strip()
    roll_no = (request.form.get("roll_no") or "").strip()
    course = (request.form.get("course") or "").strip()
    year = (request.form.get("year") or "").strip()
    section = (request.form.get("section") or "").strip()

    if not all([name, roll_no, course, year, section]):
        return jsonify({"error": "Missing required fields (name, roll_no, course, year, section)."}), 400

    required_images = [
        ("front_image", "front"),
        ("left_image", "left"),
        ("right_image", "right"),
    ]
    embeddings = []
    for field_name, label in required_images:
        if field_name not in request.files:
            return jsonify({"error": f"Missing {label} face image."}), 400
        file = request.files[field_name]
        if not file or file.filename == "":
            return jsonify({"error": f"Missing {label} face image."}), 400
        img_bytes = file.read()
        if not img_bytes:
            return jsonify({"error": f"Empty {label} face image."}), 400
        npimg = np.frombuffer(img_bytes, np.uint8)
        image_bgr = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        if image_bgr is None:
            return jsonify({"error": f"Invalid {label} face image."}), 400

        emb, err = extract_face_embedding(image_bgr)
        if err == "no_face":
            return jsonify({"error": f"No face detected in {label} image."}), 400
        if err == "invalid_image":
            return jsonify({"error": f"Invalid {label} image."}), 400
        if err == "encode_error" or emb is None:
            return jsonify(
                {"error": f"Could not compute face features for {label} image. Try again with clearer lighting."}
            ), 400

        embeddings.append(emb)

    matrix = _as_embedding_matrix(known_embeddings)
    n_emb = int(matrix.shape[0])
    n_names = len(known_names)
    emb_dim = (
        int(matrix.shape[1])
        if matrix.ndim == 2 and n_emb > 0
        else int(np.asarray(embeddings[0], dtype=np.float32).reshape(-1).shape[0])
    )
    m = min(n_emb, n_names)
    if n_emb != n_names:
        print(f"[register] Aligning embeddings.pkl: {n_emb} vectors vs {n_names} names -> {m} pairs.")
    if m == 0:
        matrix = np.empty((0, emb_dim), dtype=np.float32)
        names_base = []
    else:
        matrix = np.asarray(matrix[:m], dtype=np.float32)
        names_base = list(known_names[:m])

    if name in names_base:
        return jsonify({"error": "This student is already enrolled for recognition."}), 400

    ensure_students_schema()
    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT name, role FROM students WHERE roll_no = ?", (roll_no,))
        existing = cursor.fetchone()
        if existing is None:
            cursor.execute(
                "INSERT INTO students (name, roll_no, course, year, section, role) VALUES (?,?,?,?,?,?)",
                (name, roll_no, course, year, section, "student"),
            )
        else:
            if existing[1] and existing[1] != "student":
                conn.close()
                return jsonify({"error": "This roll number is not allowed for student registration."}), 403
            cursor.execute(
                "UPDATE students SET name = ?, course = ?, year = ?, section = ?, role = ? WHERE roll_no = ?",
                (name, course, year, section, "student", roll_no),
            )
        conn.commit()
    except sqlite3.IntegrityError as exc:
        conn.rollback()
        conn.close()
        return jsonify({"error": f"Database error: {exc!s}"}), 400
    conn.close()

    rows = [np.asarray(e, dtype=np.float32).reshape(1, -1) for e in embeddings]
    for i, row in enumerate(rows):
        if row.shape[1] != emb_dim:
            return jsonify(
                {"error": f"Internal error: embedding dimension {row.shape[1]} != {emb_dim} for angle {i + 1}."}
            ), 400
    if matrix.shape[1] != emb_dim and matrix.shape[0] > 0:
        return jsonify({"error": "Stored embeddings use a different dimension than the face model; reset embeddings.pkl or fix the file."}), 400
    if matrix.size == 0:
        matrix = np.empty((0, emb_dim), dtype=np.float32)

    try:
        if names_base:
            new_matrix = np.vstack([matrix] + rows)
        else:
            new_matrix = np.vstack(rows)
    except ValueError as exc:
        return jsonify({"error": f"Embedding shape error (corrupt model output?): {exc!s}"}), 400

    new_names = names_base + [name, name, name]

    try:
        persist_embeddings_file(new_matrix, new_names)
    except OSError:
        conn = sqlite3.connect("attendance.db")
        conn.execute("DELETE FROM students WHERE roll_no = ?", (roll_no,))
        conn.commit()
        conn.close()
        return jsonify({"error": "Failed to save embeddings file (disk full or file in use?)."}), 500
    except Exception as exc:
        conn = sqlite3.connect("attendance.db")
        conn.execute("DELETE FROM students WHERE roll_no = ?", (roll_no,))
        conn.commit()
        conn.close()
        return jsonify({"error": f"Failed to save embeddings: {exc!s}"}), 500

    known_embeddings = new_matrix
    known_names = new_names

    return jsonify({"ok": True, "message": "Student registered successfully with 3 face embeddings."})


@app.route("/facetrack-student-register", methods=["POST"])
def facetrack_student_register():
    """Same behavior as POST /register — unique path avoids proxies or other apps that intercept /register."""
    return register()


@app.route("/attendance", methods=["GET"])
def get_attendance():
    ensure_attendance_schema()
    date = (request.args.get("date") or "").strip()

    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()
    if date:
        cursor.execute(
            "SELECT name, date, time, status FROM attendance WHERE date = ? ORDER BY rowid DESC",
            (date,),
        )
    else:
        cursor.execute(
            "SELECT name, date, time, status FROM attendance ORDER BY rowid DESC"
        )
    rows = cursor.fetchall()
    conn.close()
    return jsonify(
        [{"name": r[0], "date": r[1], "time": r[2], "status": r[3]} for r in rows]
    )


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


@app.route("/recognize-multi", methods=["POST"])
def recognize_multi():
    """
    Multi-face recognition endpoint for classroom attendance.
    Returns { recognized_students: [...] } with no duplicates.
    Also marks attendance for recognized students (one per student per day).
    """
    try:
        if "image" not in request.files:
            return jsonify({"recognized_students": [], "message": "No image provided."}), 400

        file = request.files["image"]
        if not file or file.filename == "":
            return jsonify({"recognized_students": [], "message": "No image selected."}), 400

        img_bytes = file.read()
        if not img_bytes:
            return jsonify({"recognized_students": [], "message": "Empty image."}), 400

        npimg = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"recognized_students": [], "message": "Invalid image."}), 400

        recognized_faces = recognize_faces_with_boxes(frame)
        recognized = [f["name"] for f in recognized_faces]
        for name in recognized:
            mark_attendance_with_status(name=name, status=PRESENT_STATUS)

        if not recognized:
            return jsonify({"recognized_students": [], "message": "No registered faces detected."})

        return jsonify(
            {
                "recognized_students": recognized,
                "recognized_faces": recognized_faces,
                "message": "Faces recognized.",
            }
        )
    except Exception as exc:
        print("[recognize-multi]", repr(exc))
        return jsonify({"recognized_students": [], "message": "Recognition failed."}), 500


def _issue_session(email: str) -> str:
    token = secrets.token_urlsafe(32)
    exp = int(time.time()) + SESSION_DAYS * 24 * 3600
    conn = sqlite3.connect(AUTH_DB)
    # teacher session; include user data for /auth/me
    cur = conn.cursor()
    cur.execute("SELECT name, role FROM users WHERE email = ?", (email,))
    u = cur.fetchone()
    name = u[0] if u else ""
    role = u[1] if u else "Teacher"
    conn.execute(
        "INSERT INTO sessions (token, email, roll_no, role, name, expires_unix) VALUES (?,?,?,?,?,?)",
        (token, email, None, role, name, exp),
    )
    conn.commit()
    conn.close()
    return token


@app.route("/auth/register", methods=["POST"])
def auth_register():
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    email = _normalize_email(data.get("email") or "")
    password = data.get("password") or ""
    role = "Teacher"

    if not name or len(name) > 200:
        return jsonify({"error": "Name is required."}), 400
    if not _EMAIL_RE.match(email):
        return jsonify({"error": "Valid email is required."}), 400
    if len(password) < 8:
        return jsonify({"error": "Password must be at least 8 characters."}), 400
    # Admin removed (requested). Only Teacher accounts can be created here.

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


@app.route("/attendance/session/notify", methods=["POST"])
def attendance_session_notify():
    user, auth_err = _require_auth_user()
    if auth_err:
        return auth_err

    if user.get("role") == "Student" or user.get("role") not in ("Teacher", "Admin") or not user.get("email"):
        return jsonify({"error": "Only signed-in teachers can send attendance email."}), 403

    data = request.get_json(silent=True) or {}
    date = (data.get("date") or "").strip()
    if not date:
        date = datetime.now().strftime("%Y-%m-%d")
    time_label = datetime.now().strftime("%H:%M:%S")

    students = data.get("students")
    if not isinstance(students, list):
        return jsonify({"error": "students must be a list"}), 400

    verify_url = (data.get("verify_url") or "").strip()
    if not verify_url:
        verify_url = "Open Face Recognition page in the dashboard to verify."

    # Keep email content stable and readable.
    students = [str(s).strip() for s in students if str(s).strip()]

    to_email = user.get("email")
    if not to_email:
        return jsonify({"error": "No email on file for this account."}), 400

    try:
        _send_attendance_email(
            to_email=to_email,
            students=students,
            verify_url=verify_url,
            date=date,
            time_label=time_label,
        )
    except Exception as exc:
        print("[attendance/session/notify]", repr(exc))
        return jsonify({"error": "Failed to send email."}), 500

    return jsonify({"ok": True, "message": "Email sent."})


@app.route("/update-attendance", methods=["POST"])
def update_attendance():
    user, auth_err = _require_auth_user()
    if auth_err:
        return auth_err

    if user.get("role") not in ("Teacher", "Admin"):
        return jsonify({"error": "Only teachers can verify attendance."}), 403

    data = request.get_json(silent=True) or {}
    date = (data.get("date") or "").strip()
    if not date:
        date = datetime.now().strftime("%Y-%m-%d")

    verified_names = data.get("verified_names")
    if not isinstance(verified_names, list):
        return jsonify({"error": "verified_names must be a list"}), 400

    verified_names = [str(s).strip() for s in verified_names if str(s).strip()]

    ensure_attendance_schema()
    now = datetime.now()
    time = now.strftime("%H:%M:%S")

    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()

    # Remove rows not included in the final verified list.
    if verified_names:
        placeholders = ",".join(["?"] * len(verified_names))
        cursor.execute(
            f"DELETE FROM attendance WHERE date = ? AND name NOT IN ({placeholders})",
            [date] + verified_names,
        )
    else:
        cursor.execute("DELETE FROM attendance WHERE date = ?", (date,))

    # Upsert verified names with appropriate status.
    for name in verified_names:
        cursor.execute(
            "SELECT 1 FROM attendance WHERE date = ? AND name = ?",
            (date, name),
        )
        exists = cursor.fetchone() is not None
        new_status = VERIFIED_STATUS if exists else EDITED_STATUS
        if exists:
            cursor.execute(
                "UPDATE attendance SET time = ?, status = ? WHERE date = ? AND name = ?",
                (time, new_status, date, name),
            )
        else:
            cursor.execute(
                "INSERT INTO attendance (name, date, time, status) VALUES (?,?,?,?)",
                (name, date, time, new_status),
            )

    conn.commit()
    conn.close()
    return jsonify({"ok": True, "message": "Attendance updated."})


if __name__ == "__main__":
    # 0.0.0.0 so the API works when the React app is opened via your LAN IP (e.g. http://172.16.x.x:8080).
    print("Face Attendance API → http://127.0.0.1:5000 (and http://<your-LAN-IP>:5000 on this machine)")
    app.run(debug=True, host="0.0.0.0", port=5000, use_reloader=False)