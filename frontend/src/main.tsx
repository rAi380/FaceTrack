import { Component, type ErrorInfo, type ReactNode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App.tsx";
import "./index.css";

class RootErrorBoundary extends Component<{ children: ReactNode }, { err: Error | null }> {
  state = { err: null as Error | null };

  static getDerivedStateFromError(err: Error) {
    return { err };
  }

  componentDidCatch(err: Error, info: ErrorInfo) {
    console.error(err, info.componentStack);
  }

  render() {
    if (this.state.err) {
      return (
        <div style={{ fontFamily: "system-ui", padding: 24, maxWidth: 560 }}>
          <h1 style={{ fontSize: 18 }}>Something went wrong</h1>
          <p style={{ color: "#666", fontSize: 14 }}>
            The app hit a runtime error. If you recently changed login data, try clearing site data for this site or open{" "}
            <a href="/login">/login</a> again.
          </p>
          <pre
            style={{
              marginTop: 16,
              padding: 12,
              background: "#f5f5f5",
              fontSize: 12,
              overflow: "auto",
            }}
          >
            {this.state.err.message}
          </pre>
        </div>
      );
    }
    return this.props.children;
  }
}

const el = document.getElementById("root");
if (!el) {
  throw new Error("Missing #root");
}

createRoot(el).render(
  <RootErrorBoundary>
    <App />
  </RootErrorBoundary>
);
