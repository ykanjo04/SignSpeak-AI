"use client";

import { useEffect, useRef, useState } from "react";
import { backendWsBase } from "@/lib/config";
import type { PredictionMessage, Language } from "@/lib/types";
import HandSkeletonOverlay from "./HandSkeletonOverlay";

const TARGET_FPS = 15;
const JPEG_QUALITY = 0.7;
const VIDEO_W = 640;
const VIDEO_H = 480;

interface Props {
  language: Language;
  onPrediction(p: PredictionMessage): void;
}

export default function WebcamStream({ language, onPrediction }: Props) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const captureRef = useRef<HTMLCanvasElement | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const sendIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const [status, setStatus] = useState<"idle" | "starting" | "running" | "error">("idle");
  const [error, setError] = useState<string>("");
  const [lastPred, setLastPred] = useState<PredictionMessage | null>(null);

  useEffect(() => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ language }));
    }
  }, [language]);

  const start = async () => {
    try {
      setStatus("starting");
      setError("");
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: VIDEO_W, height: VIDEO_H, facingMode: "user" },
        audio: false,
      });
      if (!videoRef.current) return;
      videoRef.current.srcObject = stream;
      await videoRef.current.play();

      const ws = new WebSocket(`${backendWsBase()}/ws/live`);
      ws.binaryType = "arraybuffer";
      wsRef.current = ws;

      ws.onopen = () => {
        ws.send(JSON.stringify({ language }));
        setStatus("running");
        sendIntervalRef.current = setInterval(captureAndSend, 1000 / TARGET_FPS);
      };
      ws.onmessage = (evt) => {
        try {
          const data = JSON.parse(evt.data) as PredictionMessage;
          setLastPred(data);
          onPrediction(data);
        } catch {
          /* malformed message - ignore */
        }
      };
      ws.onerror = () => {
        setError("WebSocket error. Is the backend running on port 8000?");
        setStatus("error");
      };
      ws.onclose = () => {
        if (status === "running") setStatus("idle");
      };
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Webcam access denied");
      setStatus("error");
    }
  };

  const stop = () => {
    if (sendIntervalRef.current) {
      clearInterval(sendIntervalRef.current);
      sendIntervalRef.current = null;
    }
    wsRef.current?.close();
    wsRef.current = null;
    const v = videoRef.current;
    if (v && v.srcObject) {
      (v.srcObject as MediaStream).getTracks().forEach((t) => t.stop());
      v.srcObject = null;
    }
    setStatus("idle");
  };

  const captureAndSend = () => {
    const v = videoRef.current;
    const c = captureRef.current;
    const ws = wsRef.current;
    if (!v || !c || !ws || ws.readyState !== WebSocket.OPEN) return;
    if (v.videoWidth === 0 || v.videoHeight === 0) return;
    const ctx = c.getContext("2d");
    if (!ctx) return;
    c.width = VIDEO_W;
    c.height = VIDEO_H;
    ctx.drawImage(v, 0, 0, VIDEO_W, VIDEO_H);
    c.toBlob(
      (blob) => {
        if (!blob || ws.readyState !== WebSocket.OPEN) return;
        blob.arrayBuffer().then((ab) => ws.send(ab));
      },
      "image/jpeg",
      JPEG_QUALITY
    );
  };

  useEffect(() => {
    return () => stop();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="space-y-3">
      <div
        className="card relative overflow-hidden"
        style={{ aspectRatio: `${VIDEO_W} / ${VIDEO_H}` }}
      >
        <video
          ref={videoRef}
          muted
          playsInline
          style={{ width: "100%", height: "100%", objectFit: "cover", transform: "scaleX(-1)" }}
        />
        <HandSkeletonOverlay width={VIDEO_W} height={VIDEO_H} prediction={lastPred} />
        <canvas ref={captureRef} style={{ display: "none" }} />
        {status !== "running" && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/40">
            <button onClick={start} className="btn-primary">
              {status === "starting" ? "Starting..." : "Start camera"}
            </button>
          </div>
        )}
      </div>
      <div className="flex items-center gap-2 text-sm">
        {status === "running" && (
          <button onClick={stop} className="btn-secondary">
            Stop camera
          </button>
        )}
        {error && (
          <div className="text-rose-400">{error}</div>
        )}
      </div>
    </div>
  );
}
