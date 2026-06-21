import { useState, useRef, useCallback, useEffect } from "react";
import {
  Camera as CameraIcon,
  CheckCircle2,
  XCircle,
  Loader2,
  Video,
  VideoOff,
  Upload,
  ImageUp,
  Plus,
  X,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { toast } from "sonner";
import Camera, { type CameraHandle, type CameraOverlay } from "@/components/Camera";
import AttendanceTable from "@/components/AttendanceTable";
import {
  recognizeFace,
  recognizeMultipleFaces,
  recognizeMultipleFacesWithBoxes,
  type RecognizedFaceBox,
  fetchAttendance,
  fetchStudents,
  ApiError,
  getStoredAuthToken,
  type AttendanceRow,
  type StudentRow,
  notifyAttendanceSession,
  updateAttendance,
} from "@/services/api";

type RecognitionStatus = "idle" | "scanning" | "recognized" | "unknown";
type ScanMode = "single" | "classroom";

function mapStatusToOverlay(status: RecognitionStatus): CameraOverlay {
  if (status === "scanning") return "scanning";
  if (status === "recognized") return "recognized";
  if (status === "unknown") return "unknown";
  return "none";
}

function friendlyUnknownReason(name: string): string {
  if (name === "No Face Found") return "No face detected in the image.";
  return "Face not recognized.";
}

function todayISO(): string {
  const d = new Date();
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return `${y}-${m}-${day}`;
}

const FaceRecognitionPage = () => {
  const cameraRef = useRef<CameraHandle>(null);
  const scanLockRef = useRef(false);
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null);

  const [cameraActive, setCameraActive] = useState(false);
  const [autoMode, setAutoMode] = useState(false);
  const [status, setStatus] = useState<RecognitionStatus>("idle");
  const [detectedName, setDetectedName] = useState<string | null>(null);
  const [scanMode, setScanMode] = useState<ScanMode>("classroom");
  const [classroomSessionActive, setClassroomSessionActive] = useState(false);
  const [uiMode, setUiMode] = useState<"single" | "class_photo" | "live">("live");

  const presentSetRef = useRef<Set<string>>(new Set());
  const [presentStudents, setPresentStudents] = useState<string[]>([]);
  const [lastFrameStudents, setLastFrameStudents] = useState<string[]>([]);
  const [lastFrameFaces, setLastFrameFaces] = useState<RecognizedFaceBox[]>([]);

  const [verifyReady, setVerifyReady] = useState(false);
  const [verifyDate, setVerifyDate] = useState(todayISO());
  const [allStudents, setAllStudents] = useState<StudentRow[]>([]);
  const [studentsLoading, setStudentsLoading] = useState(false);
  const [selectedVerified, setSelectedVerified] = useState<string[]>([]);
  const [addStudentName, setAddStudentName] = useState("");

  const [attendanceRows, setAttendanceRows] = useState<AttendanceRow[]>([]);
  const [attendanceLoading, setAttendanceLoading] = useState(false);
  const [attendanceError, setAttendanceError] = useState<string | null>(null);

  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [uploadPreview, setUploadPreview] = useState<string | null>(null);
  const uploadInputRef = useRef<HTMLInputElement>(null);
  const [classPhotoFile, setClassPhotoFile] = useState<File | null>(null);
  const [classPhotoPreview, setClassPhotoPreview] = useState<string | null>(null);
  const classPhotoInputRef = useRef<HTMLInputElement>(null);
  /** Avoid duplicate attendance emails if Stop is triggered more than once for one classroom session. */
  const classroomEndEmailSentRef = useRef(false);

  const loadAttendance = useCallback(async () => {
    setAttendanceLoading(true);
    setAttendanceError(null);
    try {
      const rows = await fetchAttendance();
      setAttendanceRows(rows);
    } catch (e) {
      const msg = e instanceof ApiError ? e.message : "Could not load attendance.";
      setAttendanceError(msg);
    } finally {
      setAttendanceLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadAttendance();
  }, [loadAttendance]);

  useEffect(() => {
    if (!uploadFile) {
      setUploadPreview(null);
      return;
    }
    const url = URL.createObjectURL(uploadFile);
    setUploadPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [uploadFile]);

  useEffect(() => {
    if (!classPhotoFile) {
      setClassPhotoPreview(null);
      return;
    }
    const url = URL.createObjectURL(classPhotoFile);
    setClassPhotoPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [classPhotoFile]);

  useEffect(() => {
    if (!verifyReady) return;
    setStudentsLoading(true);
    setAllStudents([]);
    void fetchStudents()
      .then((s) => setAllStudents(s))
      .catch((e) => {
        toast.error(e instanceof ApiError ? e.message : "Could not load students list.");
      })
      .finally(() => setStudentsLoading(false));
  }, [verifyReady]);

  const handleCameraError = useCallback((message: string) => {
    toast.error(message);
    setCameraActive(false);
    setAutoMode(false);
  }, []);

  const stopCamera = useCallback(() => {
    const shouldEnd = scanMode === "classroom" && classroomSessionActive;

    setCameraActive(false);
    setAutoMode(false);
    setClassroomSessionActive(false);
    setStatus("idle");
    setDetectedName(null);
    setLastFrameFaces([]);

    if (!shouldEnd) return;

    const students = Array.from(presentSetRef.current);
    setSelectedVerified(students);
    const date = todayISO();
    setVerifyDate(date);
    setVerifyReady(true);

    const token = getStoredAuthToken();
    if (!token) {
      void loadAttendance();
      return;
    }

    if (classroomEndEmailSentRef.current) {
      void loadAttendance();
      return;
    }
    classroomEndEmailSentRef.current = true;

    const verifyUrl = window.location.origin + "/face-recognition";
    void notifyAttendanceSession({ date, students, verifyUrl })
      .then(() => toast.success("Attendance email sent to teacher."))
      .catch((e) =>
        toast.error(e instanceof ApiError ? e.message : "Attendance recorded, but email failed.")
      )
      .finally(() => {
        void loadAttendance();
      });
  }, [scanMode, classroomSessionActive, loadAttendance]);

  // Draw face structure lines (recognized face boxes) on live camera.
  useEffect(() => {
    if (!cameraActive) return;
    const canvas = overlayCanvasRef.current;
    const video = cameraRef.current?.getVideoElement();
    if (!canvas || !video) return;

    let raf = 0;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const draw = () => {
      raf = window.requestAnimationFrame(draw);
      const vw = video.videoWidth;
      const vh = video.videoHeight;
      if (!vw || !vh) return;

      const rect = video.getBoundingClientRect();
      const cw = Math.max(1, Math.floor(rect.width));
      const ch = Math.max(1, Math.floor(rect.height));
      if (canvas.width !== cw || canvas.height !== ch) {
        canvas.width = cw;
        canvas.height = ch;
      }

      ctx.clearRect(0, 0, canvas.width, canvas.height);
      if (!lastFrameFaces.length) return;

      const sx = canvas.width / vw;
      const sy = canvas.height / vh;

      ctx.lineWidth = 3;
      ctx.font = "14px ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto";
      ctx.textBaseline = "top";
      for (const f of lastFrameFaces) {
        const [x, y, w, h] = f.box;
        const rx = x * sx;
        const ry = y * sy;
        const rw = w * sx;
        const rh = h * sy;
        ctx.strokeStyle = "rgba(34,197,94,0.95)";
        ctx.strokeRect(rx, ry, rw, rh);
        const label = f.name;
        const pad = 6;
        const tw = ctx.measureText(label).width;
        ctx.fillStyle = "rgba(0,0,0,0.55)";
        ctx.fillRect(rx, Math.max(0, ry - 22), tw + pad * 2, 20);
        ctx.fillStyle = "rgba(255,255,255,0.95)";
        ctx.fillText(label, rx + pad, Math.max(0, ry - 20));
      }
    };

    raf = window.requestAnimationFrame(draw);
    return () => window.cancelAnimationFrame(raf);
  }, [cameraActive, lastFrameFaces]);

  const runRecognitionWithBlob = useCallback(
    async (blob: Blob) => {
      if (scanLockRef.current) return;
      scanLockRef.current = true;
      setStatus("scanning");
      setDetectedName(null);

      try {
        const name = await recognizeFace(blob);
        setDetectedName(name);

        const known = name !== "Unknown" && name !== "No Face Found";

        if (known) {
          setStatus("recognized");
          toast.success("Attendance marked successfully.");
        } else {
          setStatus("unknown");
          toast.error(name === "No Face Found" ? "No face detected." : "Face not recognized.");
        }

        await loadAttendance();
      } catch (e) {
        setStatus("idle");
        setDetectedName(null);
        const msg = e instanceof ApiError ? e.message : "Something went wrong. Please try again.";
        toast.error(msg);
      } finally {
        scanLockRef.current = false;
      }
    },
    [loadAttendance]
  );

  const runRecognition = useCallback(async () => {
    if (scanLockRef.current) return;
    if (!cameraRef.current) {
      toast.error("Camera is not ready.");
      return;
    }

    const blob = await cameraRef.current.captureFrame();
    if (!blob) {
      toast.error("Could not capture a frame. Wait for the video to start or check the camera.");
      return;
    }

    await runRecognitionWithBlob(blob);
  }, [runRecognitionWithBlob]);

  const runClassroomRecognitionWithBlob = useCallback(
    async (blob: Blob) => {
      if (scanLockRef.current) return;
      scanLockRef.current = true;
      setStatus("scanning");

      try {
        const { students: names, faces } = await recognizeMultipleFacesWithBoxes(blob);
        setLastFrameStudents(names);
        setLastFrameFaces(faces);

        if (names.length > 0) {
          setStatus("recognized");
          const newOnes: string[] = [];
          for (const n of names) {
            if (!presentSetRef.current.has(n)) {
              presentSetRef.current.add(n);
              newOnes.push(n);
            }
          }
          if (newOnes.length > 0) {
            setPresentStudents(Array.from(presentSetRef.current));
            toast.success(`Marked: ${newOnes.join(", ")}`);
            await loadAttendance();
          }
        } else {
          setStatus("unknown");
        }
      } catch (e) {
        setStatus("idle");
        setLastFrameFaces([]);
        const msg = e instanceof ApiError ? e.message : "Something went wrong. Please try again.";
        toast.error(msg);
      } finally {
        scanLockRef.current = false;
      }
    },
    [loadAttendance]
  );

  const runClassroomRecognition = useCallback(async () => {
    if (scanLockRef.current) return;
    if (!cameraRef.current) {
      toast.error("Camera is not ready.");
      return;
    }
    const blob = await cameraRef.current.captureFrame();
    if (!blob) {
      toast.error("Could not capture a frame. Wait for the video to start.");
      return;
    }
    await runClassroomRecognitionWithBlob(blob);
  }, [runClassroomRecognitionWithBlob]);

  const runRecognitionFromUpload = useCallback(async () => {
    if (!uploadFile) {
      toast.error("Choose a photo first.");
      return;
    }
    await runRecognitionWithBlob(uploadFile);
  }, [uploadFile, runRecognitionWithBlob]);

  const clearUpload = useCallback(() => {
    setUploadFile(null);
    setUploadPreview(null);
    if (uploadInputRef.current) uploadInputRef.current.value = "";
  }, []);

  const runClassPhotoRecognition = useCallback(async () => {
    if (!classPhotoFile) {
      toast.error("Choose a class photo first.");
      return;
    }
    setScanMode("classroom");
    setClassroomSessionActive(false);
    setStatus("scanning");
    try {
      const { students: names, faces } = await recognizeMultipleFacesWithBoxes(classPhotoFile);
      setLastFrameStudents(names);
      setLastFrameFaces(faces);
      if (names.length === 0) {
        setStatus("unknown");
        toast.error("No registered students detected in class photo.");
        return;
      }
      setStatus("recognized");
      const newOnes: string[] = [];
      for (const n of names) {
        if (!presentSetRef.current.has(n)) {
          presentSetRef.current.add(n);
          newOnes.push(n);
        }
      }
      setPresentStudents(Array.from(presentSetRef.current));
      if (newOnes.length > 0) toast.success(`Marked: ${newOnes.join(", ")}`);
      await loadAttendance();
    } catch (e) {
      setStatus("idle");
      setLastFrameFaces([]);
      toast.error(e instanceof ApiError ? e.message : "Class photo recognition failed.");
    }
  }, [classPhotoFile, loadAttendance]);

  const toggleVerifiedSelected = useCallback((name: string, next: boolean) => {
    setSelectedVerified((prev) => {
      const has = prev.includes(name);
      if (next && !has) return [...prev, name];
      if (!next && has) return prev.filter((x) => x !== name);
      return prev;
    });
  }, []);

  const addStudentToSelection = useCallback(() => {
    const raw = addStudentName.trim();
    if (!raw) return;
    const target = allStudents.find((s) => s.name.toLowerCase() === raw.toLowerCase());
    const nameToAdd = target?.name || raw;
    setSelectedVerified((prev) => (prev.includes(nameToAdd) ? prev : [...prev, nameToAdd]));
    setAddStudentName("");
  }, [addStudentName, allStudents]);

  const verifyAttendanceNow = useCallback(async () => {
    try {
      await updateAttendance({
        date: verifyDate,
        verified_names: selectedVerified,
      });
      toast.success("Attendance verified.");
      setVerifyReady(false);
      await loadAttendance();
    } catch (e) {
      toast.error(e instanceof ApiError ? e.message : "Could not verify attendance.");
    }
  }, [verifyDate, selectedVerified, loadAttendance]);

  useEffect(() => {
    if (!cameraActive || !autoMode) return;
    if (scanMode === "classroom" && !classroomSessionActive) return;
    const intervalMs = 1000; // ~1s cadence (server recognition is still the limiting factor)
    const id = window.setInterval(() => {
      if (scanMode === "classroom") {
        void runClassroomRecognition();
      } else {
        void runRecognition();
      }
    }, intervalMs);
    return () => clearInterval(id);
  }, [cameraActive, autoMode, scanMode, classroomSessionActive, runRecognition, runClassroomRecognition]);

  return (
    <div className="space-y-6 animate-fade-in max-w-7xl mx-auto px-1 sm:px-0">
      <div>
        <h1 className="page-header">Face Recognition Attendance</h1>
        <p className="text-muted-foreground text-sm mt-1">
          Use the <strong className="font-medium text-foreground">camera</strong> or{" "}
          <strong className="font-medium text-foreground">upload a photo</strong> to verify recognition and mark
          attendance. Auto mode captures from the camera every 3 seconds.
        </p>

        <div className="mt-4 flex flex-col sm:flex-row sm:items-center gap-3 sm:gap-2">
          <div className="text-sm font-medium text-foreground">Mode:</div>
          <div className="flex flex-wrap gap-2">
            <Button
              type="button"
              variant={uiMode === "class_photo" ? "default" : "outline"}
              className={uiMode === "class_photo" ? "bg-accent text-accent-foreground hover:bg-accent/90" : ""}
              onClick={() => {
                setUiMode("class_photo");
                setScanMode("classroom");
                setCameraActive(false);
                setAutoMode(false);
                setClassroomSessionActive(false);
                setVerifyReady(false);
                presentSetRef.current = new Set();
                setPresentStudents([]);
                setLastFrameStudents([]);
                setDetectedName(null);
                setStatus("idle");
              }}
            >
              Mode 1: Class Photo
            </Button>
            <Button
              type="button"
              variant={uiMode === "live" ? "default" : "outline"}
              className={uiMode === "live" ? "bg-accent text-accent-foreground hover:bg-accent/90" : ""}
              onClick={() => {
                setUiMode("live");
                setScanMode("classroom");
                setCameraActive(false);
                setAutoMode(false);
                setClassroomSessionActive(false);
                setVerifyReady(false);
                presentSetRef.current = new Set();
                setPresentStudents([]);
                setLastFrameStudents([]);
                setDetectedName(null);
                setStatus("idle");
              }}
            >
              Mode 2: Live Camera
            </Button>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 stat-card flex flex-col gap-4">
          <div className="relative">
            <Camera
              ref={cameraRef}
              active={cameraActive}
              overlay={mapStatusToOverlay(status)}
              onError={handleCameraError}
            />
            <canvas ref={overlayCanvasRef} className="absolute inset-0 w-full h-full pointer-events-none" />
          </div>

          <div className="flex flex-col sm:flex-row sm:flex-wrap gap-3 sm:items-center sm:justify-between">
            <div className="flex flex-wrap gap-3 justify-center sm:justify-start">
              {!cameraActive ? (
                <Button
                  onClick={() => {
                    if (uiMode === "class_photo") return;
                    setScanMode("classroom");
                    setCameraActive(true);
                    setClassroomSessionActive(false);
                    setStatus("idle");
                    setDetectedName(null);
                  }}
                  disabled={uiMode === "class_photo"}
                  className="bg-accent text-accent-foreground hover:bg-accent/90 w-full sm:w-auto"
                >
                  <Video className="h-4 w-4 mr-2" /> Start Camera
                </Button>
              ) : (
                <>
                  <Button onClick={stopCamera} variant="outline" className="w-full sm:w-auto">
                    <VideoOff className="h-4 w-4 mr-2" /> Stop Camera
                  </Button>
                  {uiMode === "single" && (
                    <Button
                      onClick={() => {
                        setScanMode("single");
                        setClassroomSessionActive(false);
                        setLastFrameStudents([]);
                        setAutoMode(false);
                        void runRecognition();
                      }}
                      disabled={status === "scanning"}
                      className="bg-accent text-accent-foreground hover:bg-accent/90 w-full sm:w-auto"
                    >
                      {status === "scanning" ? (
                        <>
                          <Loader2 className="h-4 w-4 mr-2 animate-spin" /> Recognizing…
                        </>
                      ) : (
                        <>
                          <CameraIcon className="h-4 w-4 mr-2" /> Start Recognition
                        </>
                      )}
                    </Button>
                  )}

                  {uiMode === "live" && (
                    <Button
                      onClick={() => {
                        presentSetRef.current = new Set();
                        setPresentStudents([]);
                        setLastFrameStudents([]);
                        setDetectedName(null);
                        setScanMode("classroom");
                        classroomEndEmailSentRef.current = false;
                        setClassroomSessionActive(true);
                        setVerifyReady(false);
                        setStatus("idle");
                        setAutoMode(true);
                      }}
                      disabled={status === "scanning"}
                      variant="secondary"
                      className="w-full sm:w-auto"
                    >
                      Start Classroom Attendance
                    </Button>
                  )}
                </>
              )}
            </div>

            {cameraActive && (
              <div className="flex items-center justify-center sm:justify-end gap-3 rounded-lg border border-border bg-muted/30 px-4 py-2">
                <Label htmlFor="auto-mode" className="text-sm cursor-pointer">
                  Auto scan ({scanMode === "classroom" ? "~2.5s" : "3s"})
                </Label>
                <Switch
                  id="auto-mode"
                  checked={autoMode}
                  onCheckedChange={setAutoMode}
                  disabled={status === "scanning"}
                />
              </div>
            )}
          </div>

          {scanMode === "classroom" && (
            <div className="rounded-xl border border-border bg-muted/20 p-4 space-y-3">
              <div className="flex items-center justify-between gap-3">
                <h3 className="text-sm font-semibold text-foreground">Students Present Today:</h3>
                <span className="text-xs text-muted-foreground">{presentStudents.length} detected</span>
              </div>
              {presentStudents.length === 0 ? (
                <p className="text-sm text-muted-foreground">No students detected yet. Click Start Classroom Attendance to begin.</p>
              ) : (
                <div className="flex flex-wrap gap-2">
                  {presentStudents.map((n) => (
                    <span
                      key={n}
                      className="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-success/10 text-success"
                    >
                      <CheckCircle2 className="h-3 w-3 mr-1" /> {n}
                    </span>
                  ))}
                </div>
              )}
            </div>
          )}

          <div className="rounded-xl border border-border bg-muted/20 p-4 space-y-3">
            <div className="flex items-center gap-2">
              <ImageUp className="h-5 w-5 text-accent shrink-0" />
              <h3 className="text-sm font-semibold text-foreground">Class photo multi-face (Mode 1)</h3>
            </div>
            <p className="text-xs text-muted-foreground">
              Upload one classroom photo to detect and mark multiple students in one shot using <code className="text-[11px] bg-muted px-1 rounded">/recognize-multi</code>.
            </p>
            <div className="space-y-2">
              <Label htmlFor="class-photo">Class photo</Label>
              <Input
                ref={classPhotoInputRef}
                id="class-photo"
                type="file"
                accept="image/*"
                className="cursor-pointer"
                disabled={status === "scanning"}
                onChange={(e) => setClassPhotoFile(e.target.files?.[0] ?? null)}
              />
            </div>
            {classPhotoPreview && (
              <div className="rounded-lg border border-border overflow-hidden max-h-40 w-full max-w-xs bg-background">
                <img src={classPhotoPreview} alt="Class preview" className="w-full h-full object-cover object-top" />
              </div>
            )}
            <div className="flex flex-wrap gap-2">
              <Button
                type="button"
                variant="secondary"
                disabled={!classPhotoFile || status === "scanning"}
                onClick={() => void runClassPhotoRecognition()}
              >
                Detect Class Photo
              </Button>
              <Button
                type="button"
                variant="outline"
                disabled={!classPhotoFile || status === "scanning"}
                onClick={() => {
                  setClassPhotoFile(null);
                  setClassPhotoPreview(null);
                  if (classPhotoInputRef.current) classPhotoInputRef.current.value = "";
                }}
              >
                Clear
              </Button>
            </div>
          </div>
        </div>

        <div className="stat-card space-y-4">
          <h3 className="section-title">
            {scanMode === "classroom" ? "Classroom Attendance" : "Recognition Result"}
          </h3>

          {scanMode === "classroom" ? (
            <>
              {!classroomSessionActive && status === "idle" && (
                <p className="text-sm text-muted-foreground">
                  Click <strong className="font-medium text-foreground">Start Classroom Attendance</strong> to begin auto detection.
                </p>
              )}

              {status === "scanning" && (
                <div className="flex flex-col items-center gap-3 py-8">
                  <Loader2 className="h-10 w-10 animate-spin text-accent" />
                  <p className="text-sm text-muted-foreground text-center px-2">
                    Scanning classroom… (multi-face)
                  </p>
                </div>
              )}

              {status === "recognized" && lastFrameStudents.length > 0 && (
                <div className="flex flex-col items-center gap-3 py-4">
                  <div className="h-16 w-16 rounded-full bg-success/20 flex items-center justify-center">
                    <CheckCircle2 className="h-8 w-8 text-success" />
                  </div>
                  <p className="text-center text-lg font-semibold text-foreground">Detected:</p>
                  <div className="flex flex-wrap gap-2 justify-center">
                    {lastFrameStudents.map((n) => (
                      <span
                        key={n}
                        className="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-success/10 text-success"
                      >
                        {n}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {status === "unknown" && (
                <div className="flex flex-col items-center gap-3 py-4">
                  <div className="h-16 w-16 rounded-full bg-destructive/20 flex items-center justify-center">
                    <XCircle className="h-8 w-8 text-destructive" />
                  </div>
                  <p className="font-semibold text-foreground text-center">
                    No registered students detected in this frame
                  </p>
                  <p className="text-sm text-muted-foreground text-center">
                    Try better lighting or ensure faces are visible.
                  </p>
                </div>
              )}

              {verifyReady && (
                <div className="rounded-xl border border-border bg-background p-4 space-y-4">
                  <h4 className="text-sm font-semibold text-foreground">Verify attendance (teacher)</h4>
                  <p className="text-xs text-muted-foreground">
                    Date: <span className="font-medium text-foreground">{verifyDate}</span>
                  </p>

                  <div className="space-y-2">
                    <Label>Your selected students</Label>
                    {selectedVerified.length === 0 ? (
                      <p className="text-sm text-muted-foreground">No students selected.</p>
                    ) : (
                      <div className="flex flex-wrap gap-2">
                        {selectedVerified.map((n) => (
                          <span
                            key={n}
                            className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-medium bg-muted/50"
                          >
                            {n}
                            <button
                              type="button"
                              className="text-muted-foreground hover:text-destructive"
                              onClick={() => toggleVerifiedSelected(n, false)}
                            >
                              <X className="h-3 w-3" />
                            </button>
                          </span>
                        ))}
                      </div>
                    )}
                  </div>

                  <div className="space-y-2">
                    <Label>Add student by name</Label>
                    <div className="flex gap-2">
                      <Input
                        value={addStudentName}
                        onChange={(e) => setAddStudentName(e.target.value)}
                        placeholder="Type exact student name"
                        disabled={studentsLoading}
                      />
                      <Button
                        type="button"
                        onClick={addStudentToSelection}
                        disabled={!addStudentName.trim() || studentsLoading}
                      >
                        <Plus className="h-4 w-4" />
                      </Button>
                    </div>
                    <p className="text-xs text-muted-foreground">
                      {studentsLoading ? "Loading registered students…" : "Remove/add students before verifying."}
                    </p>
                  </div>

                  <div className="flex flex-wrap gap-2">
                    <Button
                      type="button"
                      className="bg-accent text-accent-foreground hover:bg-accent/90 flex-1"
                      onClick={() => void verifyAttendanceNow()}
                    >
                      Verify Attendance
                    </Button>
                    <Button type="button" variant="outline" onClick={() => setVerifyReady(false)}>
                      Cancel
                    </Button>
                  </div>
                </div>
              )}
            </>
          ) : (
            <>
              {status === "idle" && (
                <p className="text-sm text-muted-foreground">
                  Use <strong className="font-medium text-foreground">Start Recognition</strong> with the camera,{" "}
                  <strong className="font-medium text-foreground">auto mode</strong>, or{" "}
                  <strong className="font-medium text-foreground">Mark attendance (upload)</strong> below the preview.
                </p>
              )}

              {status === "scanning" && (
                <div className="flex flex-col items-center gap-3 py-8">
                  <Loader2 className="h-10 w-10 animate-spin text-accent" />
                  <p className="text-sm text-muted-foreground text-center px-2">
                    Running face embedding on the server…
                  </p>
                  <p className="text-xs text-muted-foreground text-center max-w-xs">
                    First request after starting Flask can take 1–2 minutes (TensorFlow / model load). Later requests are
                    faster.
                  </p>
                </div>
              )}

              {status === "recognized" && detectedName && (
                <div className="flex flex-col items-center gap-3 py-4">
                  <div className="h-16 w-16 rounded-full bg-success/20 flex items-center justify-center">
                    <CheckCircle2 className="h-8 w-8 text-success" />
                  </div>
                  <p className="text-center text-lg font-semibold text-foreground">
                    Detected: {detectedName}
                  </p>
                  <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium bg-success/10 text-success">
                    <CheckCircle2 className="h-3 w-3" /> Attendance marked
                  </span>
                </div>
              )}

              {status === "unknown" && (
                <div className="flex flex-col items-center gap-3 py-4">
                  <div className="h-16 w-16 rounded-full bg-destructive/20 flex items-center justify-center">
                    <XCircle className="h-8 w-8 text-destructive" />
                  </div>
                  <p className="font-semibold text-foreground text-center">
                    {detectedName === "No Face Found" ? "No face detected" : "Face not recognized"}
                  </p>
                  <p className="text-sm text-muted-foreground text-center">
                    {detectedName ? friendlyUnknownReason(detectedName) : "Try again with better lighting."}
                  </p>
                </div>
              )}
            </>
          )}
        </div>
      </div>

      <AttendanceTable
        rows={attendanceRows}
        loading={attendanceLoading}
        error={attendanceError}
        title="Attendance log (from server)"
        emptyMessage="No attendance records in the database yet."
      />
    </div>
  );
};

export default FaceRecognitionPage;
