import { QueryClientProvider } from "@tanstack/react-query";
import { QueryClient } from "@tanstack/query-core";
import { BrowserRouter, Route, Routes, Navigate } from "react-router-dom";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { Toaster } from "@/components/ui/toaster";
import { TooltipProvider } from "@/components/ui/tooltip";
import { ThemeProvider } from "@/hooks/use-theme";
import { AuthProvider, useAuth } from "@/hooks/use-auth";
import DashboardLayout from "@/components/DashboardLayout";
import DashboardPage from "@/pages/DashboardPage";
import AttendancePage from "@/pages/AttendancePage";
import StudentsPage from "@/pages/StudentsPage";
import ReportsPage from "@/pages/ReportsPage";
import FaceRecognitionPage from "@/pages/FaceRecognitionPage";
import StudentRegistrationPage from "@/pages/StudentRegistrationPage";
import ProfilePage from "@/pages/ProfilePage";
import SettingsPage from "@/pages/SettingsPage";
import LoginPage from "@/pages/LoginPage";
import SignupPage from "@/pages/SignupPage";
import NotFound from "@/pages/NotFound";

const queryClient = new QueryClient();

function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const { user, authReady } = useAuth();
  if (!authReady) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background p-4">
        <p className="text-sm text-muted-foreground">Loading session…</p>
      </div>
    );
  }
  if (!user) return <Navigate to="/login" replace />;
  return <>{children}</>;
}

function AuthRoute({ children }: { children: React.ReactNode }) {
  const { user, authReady } = useAuth();
  if (!authReady) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background p-4">
        <p className="text-sm text-muted-foreground">Loading session…</p>
      </div>
    );
  }
  if (user) return <Navigate to="/" replace />;
  return <>{children}</>;
}

const AppRoutes = () => (
  <Routes>
    <Route path="/login" element={<AuthRoute><LoginPage /></AuthRoute>} />
    <Route path="/signup" element={<AuthRoute><SignupPage /></AuthRoute>} />
    {/* Public: students register face without a teacher account */}
    <Route path="/student-registration" element={<StudentRegistrationPage />} />
    <Route path="/" element={<ProtectedRoute><DashboardLayout /></ProtectedRoute>}>
      <Route index element={<DashboardPage />} />
      <Route path="attendance" element={<AttendancePage />} />
      <Route path="students" element={<StudentsPage />} />
      <Route path="reports" element={<ReportsPage />} />
      <Route path="face-recognition" element={<FaceRecognitionPage />} />
      <Route path="profile" element={<ProfilePage />} />
      <Route path="settings" element={<SettingsPage />} />
    </Route>
    <Route path="*" element={<NotFound />} />
  </Routes>
);

const App = () => (
  <QueryClientProvider client={queryClient}>
    <ThemeProvider>
      <AuthProvider>
        <TooltipProvider>
          <Toaster />
          <Sonner />
          <BrowserRouter>
            <AppRoutes />
          </BrowserRouter>
        </TooltipProvider>
      </AuthProvider>
    </ThemeProvider>
  </QueryClientProvider>
);

export default App;
