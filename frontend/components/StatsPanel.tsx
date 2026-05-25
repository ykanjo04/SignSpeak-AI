"use client";

import type { PredictionMessage } from "@/lib/types";

interface Props {
  prediction: PredictionMessage | null;
}

export default function StatsPanel({ prediction }: Props) {
  const stats = prediction?.stats;
  const stages = stats?.per_stage_ms ?? {};

  return (
    <div className="card p-4 text-xs font-mono">
      <div className="flex items-center justify-between">
        <span className="uppercase tracking-wider text-slate-400">Performance</span>
        <span className="text-slate-300">
          {(stats?.fps ?? 0).toFixed(1)} FPS &middot;{" "}
          {(stats?.latency_ms ?? 0).toFixed(0)} ms total
        </span>
      </div>
      <div className="mt-2 grid grid-cols-4 gap-1">
        {Object.entries(stages).map(([k, v]) => (
          <div key={k} className="px-2 py-1 rounded bg-white/5">
            <div className="text-slate-400">{k}</div>
            <div className="tabular-nums">{v.toFixed(1)} ms</div>
          </div>
        ))}
      </div>
    </div>
  );
}
