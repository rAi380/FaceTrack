import { Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";
import type { AttendanceRow } from "@/services/api";

export type AttendanceTableProps = {
  rows: AttendanceRow[];
  loading?: boolean;
  error?: string | null;
  title?: string;
  emptyMessage?: string;
  className?: string;
};

const AttendanceTable = ({
  rows,
  loading = false,
  error = null,
  title = "Attendance records",
  emptyMessage = "No attendance records yet.",
  className,
}: AttendanceTableProps) => {
  const showStatus = rows.some((r) => typeof r.status === "string" && r.status.trim().length > 0);
  return (
    <div className={cn("stat-card", className)}>
      <h3 className="section-title mb-4">{title}</h3>

      {error && (
        <p className="text-sm text-destructive mb-4" role="alert">
          {error}
        </p>
      )}

      {loading && !rows.length ? (
        <div className="flex items-center justify-center gap-2 py-12 text-muted-foreground">
          <Loader2 className="h-5 w-5 animate-spin" />
          <span className="text-sm">Loading attendance…</span>
        </div>
      ) : !rows.length && !error ? (
        <p className="text-sm text-muted-foreground py-6">{emptyMessage}</p>
      ) : (
        <div className="overflow-x-auto -mx-1">
          <table className="w-full min-w-[280px]">
            <thead>
              <tr className="text-left text-sm text-muted-foreground border-b border-border">
                <th className="pb-3 pr-2 font-medium">#</th>
                <th className="pb-3 pr-2 font-medium">Name</th>
                <th className="pb-3 pr-2 font-medium">Date</th>
                <th className="pb-3 font-medium">Time</th>
                {showStatus && <th className="pb-3 pr-2 font-medium">Status</th>}
              </tr>
            </thead>
            <tbody>
              {rows.map((r, i) => (
                <tr
                  key={`${r.name}-${r.date}-${r.time}-${i}`}
                  className={cn(
                    "border-b border-border/50 last:border-0",
                    i % 2 === 0 ? "bg-muted/30" : ""
                  )}
                >
                  <td className="py-3 pr-2 text-sm">{i + 1}</td>
                  <td className="py-3 pr-2 text-sm font-medium break-words max-w-[140px] sm:max-w-none">
                    {r.name}
                  </td>
                  <td className="py-3 pr-2 text-sm text-muted-foreground whitespace-nowrap">{r.date}</td>
                  <td className="py-3 text-sm text-muted-foreground whitespace-nowrap">{r.time}</td>
                  {showStatus && (
                    <td className="py-3 text-sm">
                      <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-success/10 text-success">
                        {r.status || "Present"}
                      </span>
                    </td>
                  )}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {loading && rows.length > 0 && (
        <p className="text-xs text-muted-foreground mt-2 flex items-center gap-1">
          <Loader2 className="h-3 w-3 animate-spin" /> Refreshing…
        </p>
      )}
    </div>
  );
};

export default AttendanceTable;
