export const BACKEND_HTTP =
  process.env.NEXT_PUBLIC_BACKEND_HTTP ?? "";

export function backendWsBase(): string {
  if (process.env.NEXT_PUBLIC_BACKEND_WS) {
    return process.env.NEXT_PUBLIC_BACKEND_WS.replace(/\/$/, "");
  }
  if (typeof window !== "undefined") {
    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    return `${proto}//${window.location.host}`;
  }
  return "ws://127.0.0.1:8000";
}
