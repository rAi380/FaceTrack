import { createContext, useContext, useState, ReactNode, useEffect, useCallback } from "react";
import {
  authLogin,
  authRegister,
  authMe,
  authLogout,
  getStoredAuthToken,
  AUTH_TOKEN_STORAGE_KEY,
  type AuthUserDto,
} from "@/services/api";

export type AuthUser = AuthUserDto;

interface AuthContextType {
  user: AuthUser | null;
  /** False until we finish checking stored session token (avoid login flash). */
  authReady: boolean;
  login: (email: string, password: string) => Promise<void>;
  signup: (name: string, email: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType>({
  user: null,
  authReady: false,
  login: async () => {},
  signup: async () => {},
  logout: async () => {},
});

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<AuthUser | null>(null);
  const [authReady, setAuthReady] = useState(false);

  useEffect(() => {
    let cancelled = false;
    const token = getStoredAuthToken();
    if (!token) {
      setAuthReady(true);
      return;
    }
    authMe(token)
      .then((u) => {
        if (!cancelled) setUser(u);
      })
      .catch(() => {
        localStorage.removeItem(AUTH_TOKEN_STORAGE_KEY);
        localStorage.removeItem("auth_user");
      })
      .finally(() => {
        if (!cancelled) setAuthReady(true);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const login = useCallback(async (email: string, password: string) => {
    const { token, user: u } = await authLogin(email, password);
    localStorage.setItem(AUTH_TOKEN_STORAGE_KEY, token);
    localStorage.setItem("auth_user", JSON.stringify(u));
    setUser(u);
  }, []);

  const signup = useCallback(async (name: string, email: string, password: string) => {
    const { token, user: u } = await authRegister({ name, email, password, role: "Teacher" });
    localStorage.setItem(AUTH_TOKEN_STORAGE_KEY, token);
    localStorage.setItem("auth_user", JSON.stringify(u));
    setUser(u);
  }, []);

  const logout = useCallback(async () => {
    try {
      await authLogout();
    } finally {
      localStorage.removeItem(AUTH_TOKEN_STORAGE_KEY);
      localStorage.removeItem("auth_user");
      setUser(null);
    }
  }, []);

  return (
    <AuthContext.Provider value={{ user, authReady, login, signup, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export const useAuth = () => useContext(AuthContext);
