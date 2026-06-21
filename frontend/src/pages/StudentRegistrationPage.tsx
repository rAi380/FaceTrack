import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { CheckCircle2, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Link, useNavigate } from "react-router-dom";
import { toast } from "sonner";
import Camera, { type CameraHandle } from "@/components/Camera";
import { registerStudent, ApiError } from "@/services/api";

type Step = "front" | "left" | "right";

const courses = ["B.Tech CSE", "B.Tech ECE", "B.Sc IT", "BCA", "MBA", "MCA", "M.Tech", "B.Com", "BBA"];
const years = ["1st Year", "2nd Year", "3rd Year", "4th Year"];

function blobToFile(blob: Blob, filename: string): File {
  return new File([blob], filename, { type: blob.type || "image/jpeg" });
}

const StudentRegistrationPage = () => {
  const navigate = useNavigate();
  const cameraRef = useRef<CameraHandle>(null);

  const [cameraActive, setCameraActive] = useState(true);
  const [step, setStep] = useState<Step>("front");
  const [busy, setBusy] = useState(false);

  const [name, setName] = useState("");
  const [rollNo, setRollNo] = useState("");
  const [course, setCourse] = useState(courses[0]);
  const [year, setYear] = useState(years[0]);
  const [section, setSection] = useState("");

  const [frontBlob, setFrontBlob] = useState<Blob | null>(null);
  const [leftBlob, setLeftBlob] = useState<Blob | null>(null);
  const [rightBlob, setRightBlob] = useState<Blob | null>(null);

  const [frontPreview, setFrontPreview] = useState<string | null>(null);
  const [leftPreview, setLeftPreview] = useState<string | null>(null);
  const [rightPreview, setRightPreview] = useState<string | null>(null);

  const stepLabel = useMemo(() => {
    if (step === "front") return "Front face (look straight)";
    if (step === "left") return "Left side (turn left)";
    return "Right side (turn right)";
  }, [step]);

  useEffect(() => {
    const urls: string[] = [];
    const setPreview = (b: Blob | null, setter: (u: string | null) => void) => {
      if (!b) {
        setter(null);
        return;
      }
      const u = URL.createObjectURL(b);
      urls.push(u);
      setter(u);
    };
    setPreview(frontBlob, setFrontPreview);
    setPreview(leftBlob, setLeftPreview);
    setPreview(rightBlob, setRightPreview);
    return () => urls.forEach((u) => URL.revokeObjectURL(u));
  }, [frontBlob, leftBlob, rightBlob]);

  const captureStep = useCallback(async () => {
    if (!cameraRef.current) {
      toast.error("Camera is not ready.");
      return;
    }
    const blob = await cameraRef.current.captureFrame();
    if (!blob) {
      toast.error("Could not capture frame. Wait for the video to start.");
      return;
    }

    if (step === "front") {
      setFrontBlob(blob);
      setStep("left");
      toast.success("Front photo captured.");
    } else if (step === "left") {
      setLeftBlob(blob);
      setStep("right");
      toast.success("Left photo captured.");
    } else {
      setRightBlob(blob);
      toast.success("Right photo captured.");
    }
  }, [step]);

  const submit = useCallback(async () => {
    if (!name.trim() || !rollNo.trim() || !course.trim() || !year.trim() || !section.trim()) {
      toast.error("Please fill all fields.");
      return;
    }
    if (!frontBlob || !leftBlob || !rightBlob) {
      toast.error("Capture all 3 photos (front/left/right).");
      return;
    }

    setBusy(true);
    try {
      const msg = await registerStudent({
        name,
        roll_no: rollNo,
        course,
        year,
        section,
        role: "student",
        front_image: blobToFile(frontBlob, "front.jpg"),
        left_image: blobToFile(leftBlob, "left.jpg"),
        right_image: blobToFile(rightBlob, "right.jpg"),
      });
      toast.success(msg);
      setFrontBlob(null);
      setLeftBlob(null);
      setRightBlob(null);
      setStep("front");
      window.setTimeout(() => {
        navigate("/login", { replace: true });
      }, 900);
    } catch (e) {
      toast.error(e instanceof ApiError ? e.message : "Registration failed.");
    } finally {
      setBusy(false);
    }
  }, [name, rollNo, course, year, section, frontBlob, leftBlob, rightBlob, navigate]);

  return (
    <div className="min-h-screen bg-background p-4 md:p-8 space-y-6 animate-fade-in max-w-6xl mx-auto">
      <div>
        <h1 className="page-header">Student Face Registration</h1>
        <p className="text-sm text-muted-foreground mt-1">
          Use the live camera only: capture <b>3</b> photos (front, then left, then right).
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 stat-card space-y-4">
          <div className="relative">
            <Camera ref={cameraRef} active={cameraActive} overlay="none" onError={(m) => toast.error(m)} />
          </div>

          <div className="flex flex-col sm:flex-row gap-3 sm:items-center sm:justify-between">
            <div className="text-sm text-muted-foreground">
              Current: <span className="font-medium text-foreground">{stepLabel}</span>
            </div>
            <div className="flex gap-2">
              <Button type="button" variant="outline" onClick={() => setCameraActive((v) => !v)}>
                {cameraActive ? "Stop Camera" : "Start Camera"}
              </Button>
              <Button type="button" className="bg-accent text-accent-foreground hover:bg-accent/90" onClick={() => void captureStep()} disabled={!cameraActive || busy}>
                {busy ? (<><Loader2 className="h-4 w-4 mr-2 animate-spin" /> Working…</>) : "Capture"}
              </Button>
            </div>
          </div>
        </div>

        <div className="stat-card space-y-4">
          <h3 className="section-title">Student Details</h3>

          <div className="space-y-2">
            <Label>Name</Label>
            <Input value={name} onChange={(e) => setName(e.target.value)} placeholder="Full name" />
          </div>
          <div className="space-y-2">
            <Label>Roll Number</Label>
            <Input value={rollNo} onChange={(e) => setRollNo(e.target.value)} placeholder="2415000000" />
          </div>
          <div className="space-y-2">
            <Label>Course</Label>
            <select className="w-full h-10 rounded-md border border-input bg-background px-3 text-sm" value={course} onChange={(e) => setCourse(e.target.value)} disabled={busy}>
              {courses.map((c) => <option key={c} value={c}>{c}</option>)}
            </select>
          </div>
          <div className="space-y-2">
            <Label>Year</Label>
            <select className="w-full h-10 rounded-md border border-input bg-background px-3 text-sm" value={year} onChange={(e) => setYear(e.target.value)} disabled={busy}>
              {years.map((y) => <option key={y} value={y}>{y}</option>)}
            </select>
          </div>
          <div className="space-y-2">
            <Label>Section (type manually)</Label>
            <Input value={section} onChange={(e) => setSection(e.target.value)} placeholder="A1" />
          </div>

          <div className="space-y-2">
            <Label>Captured Photos</Label>
            <div className="grid grid-cols-3 gap-2">
              <div className="rounded-md border border-border overflow-hidden h-20 bg-muted/30 flex items-center justify-center text-xs text-muted-foreground">
                {frontPreview ? <img src={frontPreview} alt="front" className="w-full h-full object-cover" /> : "Front"}
              </div>
              <div className="rounded-md border border-border overflow-hidden h-20 bg-muted/30 flex items-center justify-center text-xs text-muted-foreground">
                {leftPreview ? <img src={leftPreview} alt="left" className="w-full h-full object-cover" /> : "Left"}
              </div>
              <div className="rounded-md border border-border overflow-hidden h-20 bg-muted/30 flex items-center justify-center text-xs text-muted-foreground">
                {rightPreview ? <img src={rightPreview} alt="right" className="w-full h-full object-cover" /> : "Right"}
              </div>
            </div>
          </div>

          <Button type="button" className="w-full bg-accent text-accent-foreground hover:bg-accent/90" onClick={() => void submit()} disabled={busy}>
            {busy ? (<><Loader2 className="h-4 w-4 mr-2 animate-spin" /> Registering…</>) : (<><CheckCircle2 className="h-4 w-4 mr-2" /> Submit Registration</>)}
          </Button>
        </div>
      </div>
    </div>
  );
};

export default StudentRegistrationPage;

