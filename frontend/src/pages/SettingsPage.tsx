import { useState } from "react";
import { Moon, Sun, Bell, Shield, Key, CheckCircle2 } from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { useTheme } from "@/hooks/use-theme";
import { Button } from "@/components/ui/button";
import { toast } from "sonner";

const SettingsPage = () => {
  const { theme, setTheme } = useTheme();
  const [emailAlerts, setEmailAlerts] = useState(true);

  const handleSave = () => {
    toast.success("Settings saved successfully", {
      icon: <CheckCircle2 className="w-4 h-4 text-green-500" />,
    });
  };

  return (
    <div className="max-w-4xl mx-auto space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
      <div>
        <h1 className="text-3xl font-bold tracking-tight text-foreground">Settings</h1>
        <p className="text-muted-foreground mt-2">Configure the application preferences to match your workflow.</p>
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        {/* Appearance Settings */}
        <Card className="border-border shadow-sm bg-card transition-shadow hover:shadow-md">
          <CardHeader>
            <div className="flex items-center gap-2">
              <Sun className="w-5 h-5 text-primary" />
              <CardTitle>Appearance</CardTitle>
            </div>
            <CardDescription>Customize the look and feel of FaceTrack.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center justify-between p-3 border border-border rounded-lg bg-background/50 hover:bg-accent/50 transition-colors">
              <div>
                <p className="font-medium text-foreground">Dark Mode</p>
                <p className="text-sm text-muted-foreground">Toggle application theme</p>
              </div>
              <button
                onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
                className="relative inline-flex h-6 w-11 items-center rounded-full bg-border transition-colors focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2 focus:ring-offset-background"
                style={{ backgroundColor: theme === "dark" ? "hsl(var(--primary))" : "hsl(var(--border))" }}
              >
                <span
                  className={`${
                    theme === "dark" ? "translate-x-6" : "translate-x-1"
                  } inline-block h-4 w-4 transform rounded-full bg-white transition-transform`}
                />
              </button>
            </div>
          </CardContent>
        </Card>

        {/* Notifications Settings */}
        <Card className="border-border shadow-sm bg-card transition-shadow hover:shadow-md">
          <CardHeader>
            <div className="flex items-center gap-2">
              <Bell className="w-5 h-5 text-primary" />
              <CardTitle>Notifications</CardTitle>
            </div>
            <CardDescription>Manage how you receive alerts and reports.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center justify-between p-3 border border-border rounded-lg bg-background/50 hover:bg-accent/50 transition-colors">
              <div>
                <p className="font-medium text-foreground">Email Alerts</p>
                <p className="text-sm text-muted-foreground">Receive daily attendance reports</p>
              </div>
              <button
                onClick={() => setEmailAlerts(!emailAlerts)}
                className="relative inline-flex h-6 w-11 items-center rounded-full bg-border transition-colors focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2 focus:ring-offset-background"
                style={{ backgroundColor: emailAlerts ? "hsl(var(--primary))" : "hsl(var(--border))" }}
              >
                <span
                  className={`${
                    emailAlerts ? "translate-x-6" : "translate-x-1"
                  } inline-block h-4 w-4 transform rounded-full bg-white transition-transform`}
                />
              </button>
            </div>
          </CardContent>
        </Card>

        {/* Security Settings */}
        <Card className="border-border shadow-sm bg-card md:col-span-2 transition-shadow hover:shadow-md">
          <CardHeader>
            <div className="flex items-center gap-2">
              <Shield className="w-5 h-5 text-primary" />
              <CardTitle>Security & Account</CardTitle>
            </div>
            <CardDescription>Protect your account and manage authentication.</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex flex-col sm:flex-row items-center justify-between gap-4 p-4 border border-border rounded-lg bg-background/50">
              <div className="flex items-center gap-3">
                <div className="h-10 w-10 rounded-full bg-primary/10 flex items-center justify-center text-primary shrink-0">
                  <Key className="h-5 w-5" />
                </div>
                <div>
                  <p className="font-medium text-foreground">Change Password</p>
                  <p className="text-sm text-muted-foreground">Update your account password</p>
                </div>
              </div>
              <Button variant="outline" onClick={() => toast("Password reset email sent (simulation).")}>
                Reset Password
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>

      <div className="flex justify-end pt-4">
        <Button onClick={handleSave} size="lg" className="shadow-lg hover:shadow-primary/25 transition-all">
          Save Preferences
        </Button>
      </div>
    </div>
  );
};

export default SettingsPage;
