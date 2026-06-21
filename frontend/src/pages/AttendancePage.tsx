import { useCallback, useEffect, useState } from "react";
import { RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { toast } from "sonner";
import AttendanceTable from "@/components/AttendanceTable";
import { fetchAttendance, ApiError, type AttendanceRow } from "@/services/api";

const AttendancePage = () => {
  const [rows, setRows] = useState<AttendanceRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async (): Promise<boolean> => {
    setLoading(true);
    setError(null);
    try {
      const data = await fetchAttendance();
      setRows(data);
      return true;
    } catch (e) {
      setRows([]);
      setError(e instanceof ApiError ? e.message : "Could not load attendance.");
      return false;
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const handleRefresh = () => {
    void load().then((ok) => {
      if (ok) toast.success("Attendance refreshed.");
      else toast.error("Could not refresh attendance.");
    });
  };

  return (
    <div className="space-y-6 animate-fade-in max-w-5xl mx-auto">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="page-header">Attendance</h1>
          <p className="text-muted-foreground text-sm mt-1">
            Records from the face recognition system (SQLite via Flask API).
          </p>
        </div>
        <Button
          type="button"
          variant="outline"
          onClick={handleRefresh}
          disabled={loading}
          className="w-full sm:w-auto shrink-0"
        >
          <RefreshCw className={`h-4 w-4 mr-2 ${loading ? "animate-spin" : ""}`} />
          Refresh
        </Button>
      </div>

      <AttendanceTable
        rows={rows}
        loading={loading}
        error={error}
        title="All attendance records"
        emptyMessage="No records yet. Use Face Recognition to mark attendance."
      />
    </div>
  );
};

export default AttendancePage;
