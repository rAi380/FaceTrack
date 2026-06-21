import { forwardRef, useCallback, useEffect, useImperativeHandle, useRef } from "react";
import { Camera as CameraIcon } from "lucide-react";
import { cn } from "@/lib/utils";

export type CameraOverlay = "none" | "scanning" | "recognized" | "unknown";

export type CameraHandle = {
  /** Captures the current video frame as a JPEG blob, or null if not ready. */
  captureFrame: () => Promise<Blob | null>;
  /** Returns the underlying video element (for overlays). */
  getVideoElement: () => HTMLVideoElement | null;
};

export type CameraProps = {
  active: boolean;
  overlay?: CameraOverlay;
  className?: string;
  videoClassName?: string;
  onError?: (message: string) => void;
};

const Camera = forwardRef<CameraHandle, CameraProps>(function Camera(
  { active, overlay = "none", className, videoClassName, onError },
  ref
) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const stopStream = useCallback(() => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
  }, []);

  useEffect(() => {
    if (!active) {
      stopStream();
      return;
    }

    let cancelled = false;

    (async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: "user",
            width: { ideal: 640 },
            height: { ideal: 480 },
          },
        });
        if (cancelled) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }
        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play().catch(() => {});
        }
      } catch (err) {
        const e = err as { name?: string; message?: string };
        const name = e?.name || "";
        const secureInfo = !window.isSecureContext
          ? "Camera access requires a secure context (HTTPS). If you opened the site using a plain LAN IP over HTTP, use `localhost` or enable HTTPS."
          : "";

        let msg = "Camera access failed. Please check browser permissions and camera availability.";
        if (name === "NotAllowedError") {
          msg = "Camera permission was denied. Allow Camera access in the browser (site settings/lock icon) and try again.";
        } else if (name === "NotFoundError") {
          msg = "No camera device was found. Check your webcam connection and try again.";
        } else if (name === "NotReadableError" || name === "TrackStartError") {
          msg = "Camera is not readable (it may be in use by another app). Close other camera apps and try again.";
        } else if (name === "OverconstrainedError") {
          msg = "Camera resolution constraints are not supported. Try again.";
        } else if (name === "SecurityError" || name === "AbortError") {
          msg = "Camera access was blocked by browser security settings. " + secureInfo;
        } else if (!window.isSecureContext) {
          msg = "Camera access requires HTTPS or localhost. " + secureInfo;
        }

        onError?.(msg);
        stopStream();
      }
    })();

    return () => {
      cancelled = true;
      stopStream();
    };
  }, [active, onError, stopStream]);

  useImperativeHandle(
    ref,
    () => ({
      captureFrame: () =>
        new Promise((resolve) => {
          const video = videoRef.current;
          if (!video || video.readyState < 2) {
            resolve(null);
            return;
          }
          const w = video.videoWidth;
          const h = video.videoHeight;
          if (!w || !h) {
            resolve(null);
            return;
          }
          const canvas = document.createElement("canvas");
          canvas.width = w;
          canvas.height = h;
          const ctx = canvas.getContext("2d");
          if (!ctx) {
            resolve(null);
            return;
          }
          ctx.drawImage(video, 0, 0, w, h);
          canvas.toBlob(
            (blob) => resolve(blob),
            "image/jpeg",
            0.92
          );
        }),
      getVideoElement: () => videoRef.current,
    }),
    []
  );

  return (
    <div
      className={cn(
        "relative w-full aspect-video rounded-xl overflow-hidden bg-muted/60 border border-border",
        className
      )}
    >
      {active ? (
        <>
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            className={cn("w-full h-full object-cover", videoClassName)}
          />
          {overlay === "scanning" && (
            <div className="absolute inset-0 flex items-center justify-center bg-foreground/30 pointer-events-none">
              <div className="h-48 w-48 border-4 border-accent rounded-2xl animate-pulse" />
            </div>
          )}
          {overlay === "recognized" && (
            <div className="absolute inset-0 flex items-center justify-center bg-success/20 pointer-events-none">
              <div className="h-48 w-48 border-4 border-success rounded-2xl" />
            </div>
          )}
          {overlay === "unknown" && (
            <div className="absolute inset-0 flex items-center justify-center bg-destructive/20 pointer-events-none">
              <div className="h-48 w-48 border-4 border-destructive rounded-2xl" />
            </div>
          )}
        </>
      ) : (
        <div className="w-full h-full flex flex-col items-center justify-center gap-3 text-muted-foreground min-h-[200px] sm:min-h-0">
          <CameraIcon className="h-12 w-12 sm:h-16 sm:w-16 opacity-40" />
          <p className="text-sm text-center px-4">Camera is off. Start the camera to begin.</p>
        </div>
      )}
    </div>
  );
});

export default Camera;
