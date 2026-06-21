// @ts-nocheck
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Label } from "@/components/ui/label";
import { Calendar } from "@/components/ui/calendar";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { CalendarIcon, Download, FileText } from "lucide-react";
import { format } from "date-fns";
import { cn } from "@/lib/utils";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import { toast } from "sonner";

const courses = ["All Courses", "B.Tech CSE", "B.Tech ECE", "B.Sc IT", "BCA", "MBA", "MCA"];
const years = ["All Years", "1st Year", "2nd Year", "3rd Year", "4th Year"];
const sections = ["All Sections", "A", "B", "C", "D"];

const monthlyData = [
  { month: "Jan", present: 92, absent: 8 },
  { month: "Feb", present: 88, absent: 12 },
  { month: "Mar", present: 95, absent: 5 },
  { month: "Apr", present: 90, absent: 10 },
  { month: "May", present: 93, absent: 7 },
  { month: "Jun", present: 87, absent: 13 },
  { month: "Jul", present: 91, absent: 9 },
  { month: "Aug", present: 94, absent: 6 },
  { month: "Sep", present: 89, absent: 11 },
  { month: "Oct", present: 96, absent: 4 },
  { month: "Nov", present: 92, absent: 8 },
  { month: "Dec", present: 90, absent: 10 },
];

const ReportsPage = () => {
  const [selectedCourse, setSelectedCourse] = useState("All Courses");
  const [selectedYear, setSelectedYear] = useState("All Years");
  const [selectedSection, setSelectedSection] = useState("All Sections");
  const [dateFrom, setDateFrom] = useState<Date>(new Date());
  const [dateTo, setDateTo] = useState<Date>(new Date());

  const exportCSV = () => toast.success("Exported attendance as CSV");
  const exportPDF = () => toast.success("Exported attendance as PDF");

  return (
    <div className="space-y-6 animate-fade-in">
      <h1 className="page-header">Reports</h1>

      {/* Filters */}
      <div className="stat-card">
        <div className="flex flex-wrap gap-4 items-end">
          <div className="space-y-2">
            <Label>Course</Label>
            <Select value={selectedCourse} onValueChange={setSelectedCourse}>
              <SelectTrigger className="w-40"><SelectValue /></SelectTrigger>
              <SelectContent>
                {courses.map((c) => <SelectItem key={c} value={c}>{c}</SelectItem>)}
              </SelectContent>
            </Select>
          </div>
          <div className="space-y-2">
            <Label>Year</Label>
            <Select value={selectedYear} onValueChange={setSelectedYear}>
              <SelectTrigger className="w-32"><SelectValue /></SelectTrigger>
              <SelectContent>
                {years.map((y) => <SelectItem key={y} value={y}>{y}</SelectItem>)}
              </SelectContent>
            </Select>
          </div>
          <div className="space-y-2">
            <Label>Section</Label>
            <Select value={selectedSection} onValueChange={setSelectedSection}>
              <SelectTrigger className="w-32"><SelectValue /></SelectTrigger>
              <SelectContent>
                {sections.map((s) => <SelectItem key={s} value={s}>{s}</SelectItem>)}
              </SelectContent>
            </Select>
          </div>
          <div className="space-y-2">
            <Label>From</Label>
            <Popover>
              <PopoverTrigger asChild>
                <Button variant="outline" className={cn("w-40 justify-start text-left font-normal")}>
                  <CalendarIcon className="mr-2 h-4 w-4" />
                  {format(dateFrom, "PP")}
                </Button>
              </PopoverTrigger>
              <PopoverContent className="w-auto p-0" align="start">
                <Calendar mode="single" selected={dateFrom} onSelect={(d) => d && setDateFrom(d)} className="pointer-events-auto" />
              </PopoverContent>
            </Popover>
          </div>
          <div className="space-y-2">
            <Label>To</Label>
            <Popover>
              <PopoverTrigger asChild>
                <Button variant="outline" className={cn("w-40 justify-start text-left font-normal")}>
                  <CalendarIcon className="mr-2 h-4 w-4" />
                  {format(dateTo, "PP")}
                </Button>
              </PopoverTrigger>
              <PopoverContent className="w-auto p-0" align="start">
                <Calendar mode="single" selected={dateTo} onSelect={(d) => d && setDateTo(d)} className="pointer-events-auto" />
              </PopoverContent>
            </Popover>
          </div>
        </div>
      </div>

      {/* Export */}
      <div className="flex gap-3">
        <Button variant="outline" onClick={exportCSV}>
          <Download className="h-4 w-4 mr-2" /> Export CSV
        </Button>
        <Button variant="outline" onClick={exportPDF}>
          <FileText className="h-4 w-4 mr-2" /> Export PDF
        </Button>
      </div>

      {/* Monthly Chart */}
      <div className="stat-card animate-chart-in">
        <h3 className="section-title mb-4">Monthly Attendance Report</h3>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={monthlyData}>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
            <XAxis dataKey="month" tick={{ fontSize: 12 }} stroke="hsl(var(--muted-foreground))" />
            <YAxis tick={{ fontSize: 12 }} stroke="hsl(var(--muted-foreground))" />
            <Tooltip />
            <Bar dataKey="present" fill="hsl(142, 71%, 45%)" radius={[4, 4, 0, 0]} name="Present %" />
            <Bar dataKey="absent" fill="hsl(0, 84%, 60%)" radius={[4, 4, 0, 0]} name="Absent %" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Summary Table */}
      <div className="stat-card">
        <h3 className="section-title mb-4">Monthly Summary</h3>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="text-left text-sm text-muted-foreground border-b border-border">
                <th className="pb-3 font-medium">Month</th>
                <th className="pb-3 font-medium">Present %</th>
                <th className="pb-3 font-medium">Absent %</th>
                <th className="pb-3 font-medium">Status</th>
              </tr>
            </thead>
            <tbody>
              {monthlyData.map((m, i) => (
                <tr key={m.month} className={`border-b border-border/50 last:border-0 ${i % 2 === 0 ? "bg-muted/30" : ""}`}>
                  <td className="py-3 text-sm font-medium">{m.month}</td>
                  <td className="py-3 text-sm text-success font-medium">{m.present}%</td>
                  <td className="py-3 text-sm text-destructive">{m.absent}%</td>
                  <td className="py-3">
                    <span className={cn(
                      "text-xs px-2 py-1 rounded-full font-medium",
                      m.present >= 90
                        ? "bg-success/10 text-success"
                        : "bg-warning/10 text-warning"
                    )}>
                      {m.present >= 90 ? "Good" : "Needs Improvement"}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default ReportsPage;
