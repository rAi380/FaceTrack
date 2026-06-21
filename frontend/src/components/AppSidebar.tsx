import { Link, useLocation } from "react-router-dom";
import {
  LayoutDashboard,
  ClipboardCheck,
  Users,
  BarChart3,
  ChevronRight,
  ScanFace,
  Camera,
  X,
  Settings,
  User,
} from "lucide-react";
import { cn } from "@/lib/utils";

const navItems = [
  { label: "Overview", path: "/", icon: LayoutDashboard },
  { label: "Face Recognition", path: "/face-recognition", icon: Camera },
  { label: "Manage Attendance", path: "/attendance", icon: ClipboardCheck },
  { label: "Students", path: "/students", icon: Users },
  { label: "Reports", path: "/reports", icon: BarChart3 },
];

const bottomNavItems = [
  { label: "Profile", path: "/profile", icon: User },
  { label: "Settings", path: "/settings", icon: Settings },
];

interface AppSidebarProps {
  collapsed: boolean;
  onToggle: () => void;
}

const AppSidebar = ({ collapsed, onToggle }: AppSidebarProps) => {
  const location = useLocation();

  return (
    <aside
      className={cn(
        "fixed top-0 left-0 z-40 h-screen bg-sidebar text-sidebar-foreground transition-all duration-300 ease-in-out flex flex-col border-r border-sidebar-border",
        collapsed ? "w-16" : "w-60"
      )}
    >
      {/* Logo + toggle */}
      <div className="h-16 flex items-center justify-between px-4 border-b border-sidebar-border">
        <div className="flex items-center gap-2">
          <div className="h-9 w-9 rounded-xl bg-gradient-to-tr from-primary to-indigo-400 flex items-center justify-center shrink-0 shadow-md">
            <ScanFace className="h-5 w-5 text-white" />
          </div>
          {!collapsed && <span className="text-xl font-extrabold premium-gradient-text tracking-tight">FaceTrack</span>}
        </div>
        {!collapsed && (
          <button
            onClick={onToggle}
            className="h-7 w-7 flex items-center justify-center rounded-md text-sidebar-muted hover:text-sidebar-foreground hover:bg-sidebar-accent transition-colors"
            aria-label="Close sidebar"
          >
            <X className="h-4 w-4" />
          </button>
        )}
      </div>

      {/* Nav */}
      <nav className="flex-1 py-4 space-y-1 px-2">
        {navItems.map((item) => {
          const active = location.pathname === item.path;
          return (
            <Link
              key={item.path}
              to={item.path}
              className={cn(
                "flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all duration-150",
                active
                  ? "bg-sidebar-accent text-sidebar-primary"
                  : "text-sidebar-muted hover:bg-sidebar-accent hover:text-sidebar-foreground"
              )}
            >
              <item.icon className="h-5 w-5 shrink-0" />
              {!collapsed && <span>{item.label}</span>}
            </Link>
          );
        })}
      </nav>

      {/* Bottom Nav */}
      <div className="py-4 space-y-1 px-2 border-t border-sidebar-border mt-auto">
        {bottomNavItems.map((item) => {
          const active = location.pathname === item.path;
          return (
            <Link
              key={item.path}
              to={item.path}
              className={cn(
                "flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all duration-150",
                active
                  ? "bg-sidebar-accent text-sidebar-primary"
                  : "text-sidebar-muted hover:bg-sidebar-accent hover:text-sidebar-foreground"
              )}
            >
              <item.icon className="h-5 w-5 shrink-0" />
              {!collapsed && <span>{item.label}</span>}
            </Link>
          );
        })}
      </div>

      {/* Expand toggle (only when collapsed) */}
      {collapsed && (
        <button
          onClick={onToggle}
          className="h-12 flex items-center justify-center border-t border-sidebar-border text-sidebar-muted hover:text-sidebar-foreground transition-colors"
        >
          <ChevronRight className="h-4 w-4" />
        </button>
      )}
    </aside>
  );
};

export default AppSidebar;
