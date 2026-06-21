import { API_URL } from "@/config/api";

/** Build fetch URL so `/api/...` always hits the site origin (never broken by relative resolution). */
function apiFetchUrl(path: string): string {
  const p = path.startsWith("/") ? path : `/${path}`;
  const base = API_URL.replace(/\/$/, "");
  if (base.startsWith("http://") || base.startsWith("https://")) {
    return `${base}${p}`;
  }
  if (typeof window !== "undefined") {
    const prefix = base.startsWith("/") ? base : `/${base}`;
    return `${window.location.origin}${prefix}${p}`;
  }
  return `${base}${p}`;
}

/** First TensorFlow / FaceNet inference can take 1–2+ minutes on CPU. */
const RECOGNIZE_TIMEOUT_MS = 180_000;

export interface AttendanceRow {
  name: string;
  date: string;
  time: string;
  status?: string;
}

export interface StudentRow {
  name: string;
  roll_no: string;
  course: string;
  year: string;
  section: string;
  role?: string;
}

export class ApiError extends Error {
  constructor(
    message: string,
    public readonly status?: number,
    public readonly cause?: unknown
  ) {
    super(message);
    this.name = "ApiError";
  }
}

function parseRecognizedPerson(data: unknown): string {
  if (!data || typeof data !== "object") {
    throw new ApiError("Invalid response from server.");
  }
  const v = (data as Record<string, unknown>).recognized_person;
  if (typeof v !== "string" || !v.trim()) {
    throw new ApiError("Invalid response from server.");
  }
  return v.trim();
}

/**
 * POST /recognize — multipart form field `image` (JPEG/PNG blob).
 */
export async function recognizeFace(imageBlob: Blob): Promise<string> {
  const form = new FormData();
  form.append("image", imageBlob, "capture.jpg");

  const controller = new AbortController();
  const timeoutId = window.setTimeout(() => controller.abort(), RECOGNIZE_TIMEOUT_MS);

  let res: Response;
  try {
    res = await fetch(`${API_URL}/recognize`, {
      method: "POST",
      body: form,
      signal: controller.signal,
    });
  } catch (e) {
    const aborted =
      (e instanceof DOMException && e.name === "AbortError") ||
      (e instanceof Error && e.name === "AbortError");
    if (aborted) {
      throw new ApiError(
        "Recognition timed out. The first run often loads TensorFlow for 1–2 minutes—try again, or check the Flask terminal for errors.",
        undefined,
        e
      );
    }
    throw new ApiError("Could not reach the server. Is the Flask API running?", undefined, e);
  } finally {
    window.clearTimeout(timeoutId);
  }

  let data: unknown;
  try {
    data = await res.json();
  } catch {
    throw new ApiError(
      res.ok ? "Invalid response from server." : `Request failed (${res.status}).`
    );
  }

  if (!res.ok) {
    const msg =
      typeof (data as Record<string, unknown>)?.error === "string"
        ? String((data as Record<string, unknown>).error)
        : `Recognition failed (${res.status}).`;
    throw new ApiError(msg, res.status);
  }

  return parseRecognizedPerson(data);
}

function parseAttendanceList(data: unknown): AttendanceRow[] {
  if (!Array.isArray(data)) {
    throw new ApiError("Invalid attendance data from server.");
  }
  return data.map((row, i) => {
    if (!row || typeof row !== "object") {
      throw new ApiError(`Invalid attendance row at index ${i}.`);
    }
    const o = row as Record<string, unknown>;
    const name = o.name;
    const date = o.date;
    const time = o.time;
    const status = typeof o.status === "string" ? o.status : undefined;
    if (typeof name !== "string" || typeof date !== "string" || typeof time !== "string") {
      throw new ApiError(`Invalid attendance row at index ${i}.`);
    }
    return { name, date, time, status };
  });
}

/**
 * GET /attendance — list of { name, date, time }.
 */
export async function fetchAttendance(): Promise<AttendanceRow[]> {
  let res: Response;
  try {
    res = await fetch(`${API_URL}/attendance`);
  } catch (e) {
    throw new ApiError("Could not reach the server. Is the Flask API running?", undefined, e);
  }

  let data: unknown;
  try {
    data = await res.json();
  } catch {
    throw new ApiError(
      res.ok ? "Invalid response from server." : `Request failed (${res.status}).`
    );
  }

  if (!res.ok) {
    throw new ApiError(`Could not load attendance (${res.status}).`, res.status);
  }

  return parseAttendanceList(data);
}

function parseStudentList(data: unknown): StudentRow[] {
  if (!Array.isArray(data)) {
    throw new ApiError("Invalid student list from server.");
  }
  return data.map((row, i) => {
    if (!row || typeof row !== "object") {
      throw new ApiError(`Invalid student row at index ${i}.`);
    }
    const o = row as Record<string, unknown>;
    const name = o.name;
    const roll_no = o.roll_no;
    const course = o.course;
    const year = o.year;
    const section = o.section;
    if (
      typeof name !== "string" ||
      typeof roll_no !== "string" ||
      typeof course !== "string" ||
      typeof year !== "string" ||
      typeof section !== "string"
    ) {
      throw new ApiError(`Invalid student row at index ${i}.`);
    }
    return { name, roll_no, course, year, section };
  });
}

/**
 * GET /students — registered students (SQLite).
 */
export async function fetchStudents(): Promise<StudentRow[]> {
  let res: Response;
  try {
    res = await fetch(`${API_URL}/students`);
  } catch (e) {
    throw new ApiError("Could not reach the server. Is the Flask API running?", undefined, e);
  }

  let data: unknown;
  try {
    data = await res.json();
  } catch {
    throw new ApiError(
      res.ok ? "Invalid response from server." : `Request failed (${res.status}).`
    );
  }

  if (!res.ok) {
    const msg =
      typeof (data as Record<string, unknown>)?.error === "string"
        ? String((data as Record<string, unknown>).error)
        : `Could not load students (${res.status}).`;
    throw new ApiError(msg, res.status);
  }

  return parseStudentList(data);
}

export type RegisterStudentPayload = {
  name: string;
  roll_no: string;
  course: string;
  year: string;
  section: string;
  role: "student";
  front_image: File;
  left_image: File;
  right_image: File;
};

/**
 * POST /register — multipart form: name, roll_no, course, year, section, role, front_image, left_image, right_image.
 */
export async function registerStudent(payload: RegisterStudentPayload): Promise<string> {
  const form = new FormData();
  form.append("name", payload.name.trim());
  form.append("roll_no", payload.roll_no.trim());
  form.append("course", payload.course.trim());
  form.append("year", payload.year.trim());
  form.append("section", payload.section.trim());
  form.append("role", payload.role);
  form.append("front_image", payload.front_image);
  form.append("left_image", payload.left_image);
  form.append("right_image", payload.right_image);

  let res: Response;
  try {
    res = await fetch(apiFetchUrl("/facetrack-student-register"), {
      method: "POST",
      body: form,
      credentials: "omit",
    });
  } catch (e) {
    throw new ApiError("Could not reach the server. Is the Flask API running?", undefined, e);
  }

  let data: unknown;
  try {
    data = await res.json();
  } catch {
    throw new ApiError(
      res.ok ? "Invalid response from server." : `Registration failed (${res.status}).`
    );
  }

  if (!res.ok) {
    let msg =
      typeof (data as Record<string, unknown>)?.error === "string"
        ? String((data as Record<string, unknown>).error)
        : `Registration failed (${res.status}).`;
    if (res.status === 401 && msg === "Unauthorized") {
      msg =
        "This response usually comes from an expired teacher session in another tab (attendance email), not from registration. Close other FaceTrack tabs, clear the site login cookie, or open this page in a private window and try again.";
    }
    throw new ApiError(msg, res.status);
  }

  const message = (data as Record<string, unknown>).message;
  if (typeof message === "string" && message.trim()) {
    return message.trim();
  }
  return "Student registered successfully.";
}

// --- Authentication (Flask: /auth/*) ---

export const AUTH_TOKEN_STORAGE_KEY = "auth_token";

export interface AuthUserDto {
  name: string;
  email: string;
  role: "Teacher" | "Admin";
}

export function getStoredAuthToken(): string | null {
  return localStorage.getItem(AUTH_TOKEN_STORAGE_KEY);
}

function parseAuthUser(data: unknown): AuthUserDto {
  if (!data || typeof data !== "object") {
    throw new ApiError("Invalid user response.");
  }
  const o = data as Record<string, unknown>;
  const name = o.name;
  const email = o.email;
  const role = o.role;
  if (typeof name !== "string" || typeof email !== "string") {
    throw new ApiError("Invalid user response.");
  }
  if (role !== "Teacher" && role !== "Admin") {
    throw new ApiError("Invalid user response.");
  }
  return { name, email, role };
}

function errorMessageFromBody(data: unknown, fallback: string): string {
  if (data && typeof data === "object" && typeof (data as Record<string, unknown>).error === "string") {
    return String((data as Record<string, unknown>).error);
  }
  return fallback;
}

export async function authRegister(payload: {
  name: string;
  email: string;
  password: string;
  role: "Teacher";
}): Promise<{ token: string; user: AuthUserDto }> {
  let res: Response;
  try {
    res = await fetch(`${API_URL}/auth/register`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  } catch (e) {
    throw new ApiError("Could not reach the server. Is the Flask API running?", undefined, e);
  }

  let data: unknown;
  try {
    data = await res.json();
  } catch {
    throw new ApiError(res.ok ? "Invalid response from server." : `Registration failed (${res.status}).`);
  }

  if (!res.ok) {
    throw new ApiError(errorMessageFromBody(data, `Registration failed (${res.status}).`), res.status);
  }

  const token = (data as Record<string, unknown>).token;
  const user = (data as Record<string, unknown>).user;
  if (typeof token !== "string" || !token) {
    throw new ApiError("Invalid response from server.");
  }
  return { token, user: parseAuthUser(user) };
}

export async function authLogin(email: string, password: string): Promise<{ token: string; user: AuthUserDto }> {
  let res: Response;
  try {
    res = await fetch(`${API_URL}/auth/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email: email.trim(), password }),
    });
  } catch (e) {
    throw new ApiError("Could not reach the server. Is the Flask API running?", undefined, e);
  }

  let data: unknown;
  try {
    data = await res.json();
  } catch {
    throw new ApiError(
      res.ok ? "Invalid response from server." : `Login failed (${res.status}).`
    );
  }

  if (!res.ok) {
    throw new ApiError(errorMessageFromBody(data, `Login failed (${res.status}).`), res.status);
  }

  const token = (data as Record<string, unknown>).token;
  const user = (data as Record<string, unknown>).user;
  if (typeof token !== "string" || !token) {
    throw new ApiError("Invalid response from server.");
  }
  return { token, user: parseAuthUser(user) };
}

export async function authMe(token: string): Promise<AuthUserDto> {
  let res: Response;
  try {
    res = await fetch(`${API_URL}/auth/me`, {
      headers: { Authorization: `Bearer ${token}` },
    });
  } catch (e) {
    throw new ApiError("Could not reach the server.", undefined, e);
  }

  let data: unknown;
  try {
    data = await res.json();
  } catch {
    throw new ApiError("Invalid response from server.");
  }

  if (!res.ok) {
    throw new ApiError("Session expired.", res.status);
  }

  const user = (data as Record<string, unknown>).user;
  return parseAuthUser(user);
}

export async function authLogout(): Promise<void> {
  const token = getStoredAuthToken();
  if (!token) return;
  try {
    await fetch(`${API_URL}/auth/logout`, {
      method: "POST",
      headers: { Authorization: `Bearer ${token}` },
    });
  } catch {
    /* still clear client */
  }
}

function requireAuthHeaders(): HeadersInit {
  const token = getStoredAuthToken();
  if (!token) {
    throw new ApiError("You are not authenticated. Please log in.");
  }
  return {
    Authorization: `Bearer ${token}`,
  };
}

function parseRecognizeMultiResponse(data: unknown): string[] {
  if (!data || typeof data !== "object") {
    throw new ApiError("Invalid response from server.");
  }
  const o = data as Record<string, unknown>;
  const arr = o.recognized_students;
  if (!Array.isArray(arr)) {
    throw new ApiError("Invalid response from server.");
  }
  return arr
    .map((x) => (typeof x === "string" ? x.trim() : ""))
    .filter((x) => x.length > 0);
}

export async function recognizeMultipleFaces(imageBlob: Blob): Promise<string[]> {
  const form = new FormData();
  form.append("image", imageBlob, "capture.jpg");

  let res: Response;
  try {
    res = await fetch(`${API_URL}/recognize-multi`, {
      method: "POST",
      body: form,
    });
  } catch (e) {
    throw new ApiError("Could not reach the server. Is the Flask API running?", undefined, e);
  }

  let data: unknown;
  try {
    data = await res.json();
  } catch {
    throw new ApiError(
      res.ok ? "Invalid response from server." : `Request failed (${res.status}).`
    );
  }

  if (!res.ok) {
    const msg =
      typeof (data as Record<string, unknown>)?.error === "string"
        ? String((data as Record<string, unknown>).error)
        : `Recognition failed (${res.status}).`;
    throw new ApiError(msg, res.status);
  }

  return parseRecognizeMultiResponse(data);
}

export type RecognizedFaceBox = {
  name: string;
  box: [number, number, number, number];
};

export async function recognizeMultipleFacesWithBoxes(imageBlob: Blob): Promise<{
  students: string[];
  faces: RecognizedFaceBox[];
}> {
  const form = new FormData();
  form.append("image", imageBlob, "capture.jpg");

  let res: Response;
  try {
    res = await fetch(`${API_URL}/recognize-multi`, {
      method: "POST",
      body: form,
    });
  } catch (e) {
    throw new ApiError("Could not reach the server. Is the Flask API running?", undefined, e);
  }

  let data: unknown;
  try {
    data = await res.json();
  } catch {
    throw new ApiError(res.ok ? "Invalid response from server." : `Request failed (${res.status}).`);
  }

  if (!res.ok) {
    const msg =
      typeof (data as Record<string, unknown>)?.error === "string"
        ? String((data as Record<string, unknown>).error)
        : `Recognition failed (${res.status}).`;
    throw new ApiError(msg, res.status);
  }

  const students = parseRecognizeMultiResponse(data);
  const facesRaw = (data as Record<string, unknown>).recognized_faces;
  const faces: RecognizedFaceBox[] = Array.isArray(facesRaw)
    ? facesRaw
        .map((x) => {
          if (!x || typeof x !== "object") return null;
          const o = x as Record<string, unknown>;
          if (typeof o.name !== "string") return null;
          const b = o.box;
          if (!Array.isArray(b) || b.length !== 4) return null;
          const nums = b.map((n) => Number(n));
          if (nums.some((n) => !Number.isFinite(n))) return null;
          return { name: o.name.trim(), box: [nums[0], nums[1], nums[2], nums[3]] as const };
        })
        .filter((v): v is RecognizedFaceBox => !!v && !!v.name)
    : [];

  return { students, faces };
}

export async function notifyAttendanceSession(payload: {
  date: string;
  students: string[];
  verifyUrl?: string;
}): Promise<void> {
  const res = await fetch(`${API_URL}/attendance/session/notify`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...requireAuthHeaders(),
    },
    body: JSON.stringify({
      date: payload.date,
      students: payload.students,
      verify_url: payload.verifyUrl,
    }),
  });

  let data: unknown;
  try {
    data = await res.json();
  } catch {
    data = {};
  }

  if (!res.ok) {
    let msg =
      typeof (data as Record<string, unknown>)?.error === "string"
        ? String((data as Record<string, unknown>).error)
        : `Email failed (${res.status}).`;
    if (res.status === 401 && msg === "Unauthorized") {
      msg =
        "Your teacher session expired. Sign in again to send the attendance email. (Attendance already saved on the server.)";
    }
    throw new ApiError(msg, res.status);
  }
}

export async function updateAttendance(payload: {
  date: string;
  verified_names: string[];
}): Promise<void> {
  const res = await fetch(`${API_URL}/update-attendance`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...requireAuthHeaders(),
    },
    body: JSON.stringify({
      date: payload.date,
      verified_names: payload.verified_names,
    }),
  });

  let data: unknown;
  try {
    data = await res.json();
  } catch {
    data = {};
  }

  if (!res.ok) {
    let msg =
      typeof (data as Record<string, unknown>)?.error === "string"
        ? String((data as Record<string, unknown>).error)
        : `Update failed (${res.status}).`;
    if (res.status === 401 && msg === "Unauthorized") {
      msg = "Your teacher session expired. Sign in again to verify attendance.";
    }
    throw new ApiError(msg, res.status);
  }
}
