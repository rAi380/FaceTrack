import { useCallback, useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { ExternalLink, Plus, Search, Loader2, RefreshCw, Info } from "lucide-react";
import { cn } from "@/lib/utils";
import { fetchStudents, ApiError, type StudentRow } from "@/services/api";

const StudentsPage = () => {
  const [students, setStudents] = useState<StudentRow[]>([]);
  const [listLoading, setListLoading] = useState(true);
  const [listError, setListError] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const [dialogOpen, setDialogOpen] = useState(false);

  const loadStudents = useCallback(async () => {
    setListLoading(true);
    setListError(null);
    try {
      const rows = await fetchStudents();
      setStudents(rows);
    } catch (e) {
      setStudents([]);
      setListError(e instanceof ApiError ? e.message : "Could not load students.");
    } finally {
      setListLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadStudents();
  }, [loadStudents]);

  const filtered = students.filter(
    (s) =>
      s.name.toLowerCase().includes(search.toLowerCase()) ||
      s.roll_no.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div className="space-y-6 animate-fade-in max-w-6xl mx-auto">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="page-header">Student Management</h1>
          <p className="text-sm text-muted-foreground mt-1">
            Students add themselves via live camera on the public page. Embeddings go to{" "}
            <code className="text-xs bg-muted px-1 rounded">embeddings.pkl</code>; this table lists who is in SQLite.
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          <Button type="button" variant="outline" onClick={() => void loadStudents()} disabled={listLoading}>
            <RefreshCw className={cn("h-4 w-4 mr-2", listLoading && "animate-spin")} />
            Refresh
          </Button>
          <Dialog open={dialogOpen} onOpenChange={setDialogOpen}>
            <DialogTrigger asChild>
              <Button type="button" className="bg-accent text-accent-foreground hover:bg-accent/90">
                <Plus className="h-4 w-4 mr-2" /> How students register
              </Button>
            </DialogTrigger>
            <DialogContent className="max-w-md">
              <DialogHeader>
                <DialogTitle className="flex items-center gap-2">
                  <Info className="h-5 w-5" />
                  Student face registration
                </DialogTitle>
              </DialogHeader>
              <div className="space-y-3 text-sm text-muted-foreground">
                <p>
                  Registration is <strong className="text-foreground">not</strong> done from the teacher dashboard.
                  Share this link with students:
                </p>
                <div className="rounded-lg border border-border bg-muted/40 p-3 font-mono text-xs break-all text-foreground">
                  {typeof window !== "undefined"
                    ? `${window.location.origin}/student-registration`
                    : "/student-registration"}
                </div>
                <p>They use the webcam only (three angles: front, left, right) and submit their details there.</p>
                <Button asChild className="w-full bg-accent text-accent-foreground hover:bg-accent/90">
                  <Link to="/student-registration" target="_blank" rel="noopener noreferrer">
                    <ExternalLink className="h-4 w-4 mr-2" />
                    Open registration page
                  </Link>
                </Button>
              </div>
            </DialogContent>
          </Dialog>
        </div>
      </div>

      {listError && (
        <p className="text-sm text-destructive rounded-lg border border-destructive/30 bg-destructive/5 px-4 py-3">{listError}</p>
      )}

      <div className="relative max-w-sm">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
        <Input
          placeholder="Search by name or roll…"
          className="pl-9"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
      </div>

      <div className="stat-card overflow-x-auto">
        {listLoading ? (
          <div className="flex items-center justify-center gap-2 py-16 text-muted-foreground">
            <Loader2 className="h-5 w-5 animate-spin" />
            <span className="text-sm">Loading students…</span>
          </div>
        ) : (
          <table className="w-full min-w-[600px]">
            <thead>
              <tr className="text-left text-sm text-muted-foreground border-b border-border">
                <th className="pb-3 font-medium">#</th>
                <th className="pb-3 font-medium">Name</th>
                <th className="pb-3 font-medium">Roll no</th>
                <th className="pb-3 font-medium">Course</th>
                <th className="pb-3 font-medium">Year</th>
                <th className="pb-3 font-medium">Section</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((s, i) => (
                <tr
                  key={s.roll_no}
                  className={cn(
                    "border-b border-border/50 last:border-0 hover:bg-muted/50 transition-colors",
                    i % 2 === 0 ? "bg-muted/30" : ""
                  )}
                >
                  <td className="py-3 text-sm">{i + 1}</td>
                  <td className="py-3 text-sm font-medium">
                    <div className="flex items-center gap-2">
                      <div className="h-8 w-8 rounded-full bg-accent/20 flex items-center justify-center text-xs font-semibold text-accent shrink-0">
                        {s.name.charAt(0).toUpperCase()}
                      </div>
                      {s.name}
                    </div>
                  </td>
                  <td className="py-3 text-sm text-muted-foreground">{s.roll_no}</td>
                  <td className="py-3 text-sm">{s.course}</td>
                  <td className="py-3 text-sm">{s.year}</td>
                  <td className="py-3 text-sm">{s.section}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        {!listLoading && filtered.length === 0 && (
          <p className="text-sm text-muted-foreground py-8 text-center">
            {students.length === 0
              ? "No students yet. Share the public registration link with your class."
              : "No matches for your search."}
          </p>
        )}
      </div>
    </div>
  );
};

export default StudentsPage;
