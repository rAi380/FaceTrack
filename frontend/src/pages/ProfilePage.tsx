import { useAuth } from "@/hooks/use-auth";
import { User, Mail, Shield, Calendar, MapPin } from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

const ProfilePage = () => {
  const { user } = useAuth();

  return (
    <div className="max-w-4xl mx-auto space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
      <div>
        <h1 className="text-3xl font-bold tracking-tight text-foreground">Profile</h1>
        <p className="text-muted-foreground mt-2">Manage your personal information and preferences.</p>
      </div>

      <div className="grid gap-6 md:grid-cols-[1fr_2fr]">
        <Card className="border-border shadow-sm bg-card hover:shadow-md transition-shadow">
          <CardContent className="p-6 flex flex-col items-center text-center space-y-4">
            <div className="h-24 w-24 rounded-full bg-primary/10 flex items-center justify-center text-primary text-3xl font-bold border-4 border-background shadow-sm">
              {user?.name?.charAt(0) || "U"}
            </div>
            <div>
              <h2 className="text-xl font-semibold text-foreground">{user?.name}</h2>
              <p className="text-sm text-muted-foreground mt-1">{user?.role || "Teacher"}</p>
            </div>
            <Badge variant="secondary" className="mt-2 capitalize bg-primary/10 text-primary hover:bg-primary/20 transition-colors">
              <Shield className="w-3 h-3 mr-1" />
              {user?.role || "Teacher"} Account
            </Badge>
          </CardContent>
        </Card>

        <Card className="border-border shadow-sm bg-card">
          <CardHeader>
            <CardTitle>Personal Details</CardTitle>
            <CardDescription>Your registered account details and contact info.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="space-y-4">
              <div className="flex items-center gap-4 p-3 rounded-lg hover:bg-accent/50 transition-colors">
                <div className="h-10 w-10 rounded-full bg-primary/10 flex items-center justify-center text-primary shrink-0">
                  <User className="h-5 w-5" />
                </div>
                <div>
                  <p className="text-sm font-medium text-muted-foreground">Full Name</p>
                  <p className="text-base font-semibold text-foreground">{user?.name || "Not set"}</p>
                </div>
              </div>

              <div className="flex items-center gap-4 p-3 rounded-lg hover:bg-accent/50 transition-colors">
                <div className="h-10 w-10 rounded-full bg-primary/10 flex items-center justify-center text-primary shrink-0">
                  <Mail className="h-5 w-5" />
                </div>
                <div>
                  <p className="text-sm font-medium text-muted-foreground">Email Address</p>
                  <p className="text-base font-semibold text-foreground">{user?.email || "Not set"}</p>
                </div>
              </div>

              <div className="flex items-center gap-4 p-3 rounded-lg hover:bg-accent/50 transition-colors">
                <div className="h-10 w-10 rounded-full bg-primary/10 flex items-center justify-center text-primary shrink-0">
                  <Calendar className="h-5 w-5" />
                </div>
                <div>
                  <p className="text-sm font-medium text-muted-foreground">Joined</p>
                  <p className="text-base font-semibold text-foreground">Recently</p>
                </div>
              </div>

              <div className="flex items-center gap-4 p-3 rounded-lg hover:bg-accent/50 transition-colors">
                <div className="h-10 w-10 rounded-full bg-primary/10 flex items-center justify-center text-primary shrink-0">
                  <MapPin className="h-5 w-5" />
                </div>
                <div>
                  <p className="text-sm font-medium text-muted-foreground">Location</p>
                  <p className="text-base font-semibold text-foreground">Local System</p>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default ProfilePage;
