"use client";

import { useState } from "react";
import { BACKEND_HTTP } from "@/lib/config";
import LanguageToggle from "@/components/LanguageToggle";
import type { Language } from "@/lib/types";

interface TranscriptEntry {
  t_s: number;
  label: string;
  display: string;
  confidence: number;
}

interface UploadResponse {
  frames_processed: number;
  duration_s: number;
  transcript: TranscriptEntry[];
  text: string;
  language: string;
}

export default function UploadPage() {
  const [language, setLanguage] = useState<Language>("asl");
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<UploadResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string>("");

  const onSubmit = async () => {
    if (!file) return;
    setBusy(true);
    setError("");
    setResult(null);
    try {
      const fd = new FormData();
      fd.append("file", file);
      fd.append("language", language);
      const r = await fetch(`${BACKEND_HTTP}/upload`, {
        method: "POST",
        body: fd,
      });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const data: UploadResponse = await r.json();
      setResult(data);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Upload failed");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="space-y-6">
      <header className="flex items-center justify-between flex-wrap gap-3">
        <div>
          <h1 className="text-3xl font-bold">Upload a video</h1>
          <p className="text-slate-400 text-sm">
            Drop an mp4 with a single signer and get back a frame-by-frame transcript.
          </p>
        </div>
        <LanguageToggle value={language} onChange={setLanguage} />
      </header>

      <div className="card p-6 space-y-4">
        <input
          type="file"
          accept="video/*"
          onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          className="block w-full text-sm file:mr-4 file:py-2 file:px-4
                     file:rounded-lg file:border-0 file:bg-blue-500/20
                     file:text-blue-200 hover:file:bg-blue-500/30"
        />
        <button onClick={onSubmit} disabled={!file || busy} className="btn-primary">
          {busy ? "Processing..." : "Transcribe"}
        </button>
        {error && <div className="text-rose-400 text-sm">{error}</div>}
      </div>

      {result && (
        <div className="card p-6 space-y-4">
          <div className="text-sm text-slate-400">
            Processed {result.frames_processed} frames in {result.duration_s} s ({result.language}).
          </div>
          <div>
            <div className="text-xs uppercase tracking-wider text-slate-400">
              Recognised text
            </div>
            <div className="font-mono text-2xl mt-1" dir="auto">
              {result.text || "(no signs detected)"}
            </div>
          </div>
          <div>
            <div className="text-xs uppercase tracking-wider text-slate-400 mb-2">
              Timeline
            </div>
            <div className="space-y-1 text-sm">
              {result.transcript.length === 0 && <div className="text-slate-500">No stable predictions</div>}
              {result.transcript.map((t, i) => (
                <div key={i} className="flex items-center gap-3">
                  <span className="tabular-nums text-slate-400 w-16">{t.t_s.toFixed(1)}s</span>
                  <span className="font-mono w-12">{t.display || t.label}</span>
                  <span className="text-slate-500">{Math.round(t.confidence * 100)}%</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
