import { useCallback, useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { Users, UserCheck, Clock, Database, ArrowRight } from "lucide-react";
import { useAuth } from "@/hooks/use-auth";
import { Button } from "@/components/ui/button";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import { fetchAttendance, ApiError, type AttendanceRow } from "@/services/api";

function todayISO() {
  const d = new Date();
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return `${y}-${m}-${day}`;
}

const DashboardPage = () => {
  const { user } = useAuth();
  const [rows, setRows] = useState<AttendanceRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await fetchAttendance();
      setRows(data);
    } catch (e) {
      setRows([]);
      setError(e instanceof ApiError ? e.message : "Could not load attendance.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const today = todayISO();

  const stats = useMemo(() => {
    const todayRows = rows.filter((r) => r.date === today);
    const uniqueToday = new Set(todayRows.map((r) => r.name)).size;
    const last = rows[0];
    return {
      total: rows.length,
      todayCount: todayRows.length,
      uniqueToday,
      lastTime: last ? `${last.date} ${last.time}` : "—",
    };
  }, [rows, today]);

  const barData = useMemo(() => {
    const days = 14;
    const map = new Map<string, number>();
    const start = new Date();
    start.setDate(start.getDate() - (days - 1));
    for (let i = 0; i < days; i++) {
      const d = new Date(start);
      d.setDate(start.getDate() + i);
      const y = d.getFullYear();
      const m = String(d.getMonth() + 1).padStart(2, "0");
      const day = String(d.getDate()).padStart(2, "0");
      map.set(`${y}-${m}-${day}`, 0);
    }
    for (const r of rows) {
      if (map.has(r.date)) map.set(r.date, (map.get(r.date) || 0) + 1);
    }
    return Array.from(map.entries()).map(([date, count]) => ({
      label: date.slice(5),
      count,
    }));
  }, [rows]);

  const recent = rows.slice(0, 6);

  return (
    <div className="space-y-6 animate-fade-in max-w-7xl mx-auto">
      <div>
        <h1 className="page-header">Hello, {user?.name || "User"}</h1>
        <p className="text-muted-foreground text-sm mt-1">
          Face attendance overview from the live API. Open{" "}
          <Link to="/face-recognition" className="text-accent underline-offset-4 hover:underline font-medium">
            Face Recognition
          </Link>{" "}
          to mark attendance.
        </p>
      </div>

      {error && (
        <p className="text-sm text-destructive rounded-lg border border-destructive/30 bg-destructive/5 px-4 py-3">
          {error}
        </p>
      )}

      <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4">
        <div className="stat-card flex items-center gap-4">
          <div className="h-12 w-12 rounded-xl bg-primary/10 flex items-center justify-center">
            <Database className="h-6 w-6 text-foreground" />
          </div>
          <div>
            <p className="text-sm text-muted-foreground">Total scans (stored)</p>
            <p className="text-2xl font-semibold text-foreground">
              {loading ? "…" : stats.total}
            </p>
          </div>
        </div>
        <div className="stat-card flex items-center gap-4">
          <div className="h-12 w-12 rounded-xl bg-success/10 flex items-center justify-center">
            <UserCheck className="h-6 w-6 text-success" />
          </div>
          <div>
            <p className="text-sm text-muted-foreground">Check-ins today</p>
            <p className="text-2xl font-semibold text-foreground">
              {loading ? "…" : stats.todayCount}
            </p>
          </div>
        </div>
        <div className="stat-card flex items-center gap-4">
          <div className="h-12 w-12 rounded-xl bg-accent/10 flex items-center justify-center">
            <Users className="h-6 w-6 text-accent" />
          </div>
          <div>
            <p className="text-sm text-muted-foreground">Unique people today</p>
            <p className="text-2xl font-semibold text-foreground">
              {loading ? "…" : stats.uniqueToday}
            </p>
          </div>
        </div>
        <div className="stat-card flex items-center gap-4">
          <div className="h-12 w-12 rounded-xl bg-warning/10 flex items-center justify-center">
            <Clock className="h-6 w-6 text-warning" />
          </div>
          <div className="min-w-0">
            <p className="text-sm text-muted-foreground">Latest record</p>
            <p className="text-lg font-semibold text-foreground truncate" title={stats.lastTime}>
              {loading ? "…" : stats.lastTime}
            </p>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="stat-card animate-chart-in">
          <h3 className="section-title mb-4">Check-ins per day (last 14 days)</h3>
          <div className="h-[260px] w-full min-h-[200px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={barData}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                <XAxis dataKey="label" tick={{ fontSize: 10 }} stroke="hsl(var(--muted-foreground))" />
                <YAxis allowDecimals={false} tick={{ fontSize: 11 }} stroke="hsl(var(--muted-foreground))" />
                <Tooltip />
                <Bar dataKey="count" fill="hsl(222, 47%, 11%)" radius={[6, 6, 0, 0]} name="Check-ins" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="stat-card flex flex-col gap-4">
          <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-3">
            <div>
              <h3 className="section-title">Mark attendance</h3>
              <p className="text-sm text-muted-foreground mt-1">
                Start single scan, class photo, or live CCTV on the Face Recognition page.
              </p>
            </div>
            <Button asChild variant="outline" size="sm" className="shrink-0">
              <Link to="/face-recognition">
                Go to Face Recognition <ArrowRight className="h-4 w-4 ml-1" />
              </Link>
            </Button>
          </div>
        </div>
      </div>

      <div className="stat-card">
        <h3 className="section-title mb-4">Recent check-ins</h3>
        {!recent.length && !loading ? (
          <p className="text-sm text-muted-foreground">No records yet. Mark attendance from Face Recognition.</p>
        ) : (
          <ul className="space-y-2">
            {recent.map((r, i) => (
              <li
                key={`${r.name}-${r.date}-${r.time}-${i}`}
                className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-1 rounded-lg border border-border/60 px-3 py-2 text-sm"
              >
                <span className="font-medium text-foreground">{r.name}</span>
                <span className="text-muted-foreground tabular-nums">
                  {r.date} · {r.time}
                </span>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
};

export default DashboardPage;
