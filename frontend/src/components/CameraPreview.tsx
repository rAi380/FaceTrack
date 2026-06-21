import { useState, useCallback } from "react";
import { Video, VideoOff } from "lucide-react";
import { Button } from "@/components/ui/button";
import { toast } from "sonner";
import Camera from "@/components/Camera";

/**
 * Dashboard-only live camera preview (no capture / recognize).
 */
const CameraPreview = () => {
  const [active, setActive] = useState(false);

  const onError = useCallback((message: string) => {
    toast.error(message);
    setActive(false);
  }, []);

  return (
    <div className="space-y-3 w-full">
      <Camera active={active} overlay="none" onError={onError} className="max-h-[280px]" />
      <div className="flex flex-wrap justify-center gap-2">
        {!active ? (
          <Button
            type="button"
            onClick={() => setActive(true)}
            size="sm"
            className="bg-accent text-accent-foreground hover:bg-accent/90"
          >
            <Video className="h-4 w-4 mr-2" /> Start preview
          </Button>
        ) : (
          <Button type="button" onClick={() => setActive(false)} variant="outline" size="sm">
            <VideoOff className="h-4 w-4 mr-2" /> Stop preview
          </Button>
        )}
      </div>
    </div>
  );
};

export default CameraPreview;
