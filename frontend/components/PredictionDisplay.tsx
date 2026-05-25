"use client";

import type { PredictionMessage } from "@/lib/types";

interface Props {
  prediction: PredictionMessage | null;
}

export default function PredictionDisplay({ prediction }: Props) {
  const label = prediction?.label_id != null && prediction.label_id >= 0 ? prediction.label : "...";
  const conf = prediction?.confidence ?? 0;
  const motion = prediction?.motion ?? "";

  return (
    <div className="card p-6">
      <div className="text-xs uppercase tracking-wider text-slate-400">
        Detected sign
      </div>
      <div className="text-6xl font-extrabold mt-1">{label}</div>
      <div className="mt-2 flex items-center gap-3">
        <div className="h-2 flex-1 bg-white/10 rounded-full overflow-hidden">
          <div
            className="h-full bg-gradient-to-r from-blue-500 to-purple-500"
            style={{ width: `${Math.round(conf * 100)}%` }}
          />
        </div>
        <div className="text-sm tabular-nums">{Math.round(conf * 100)}%</div>
      </div>
      {motion && (
        <div className="mt-2 text-xs text-slate-400">
          motion: <span className="text-slate-200">{motion}</span>
          {prediction?.per_model && Object.keys(prediction.per_model).length > 0 && (
            <>
              {" "}&middot;{" "}
              {Object.entries(prediction.per_model)
                .map(([k, v]) => `${k}=${(v * 100).toFixed(0)}%`)
                .join("  ")}
            </>
          )}
        </div>
      )}
    </div>
  );
}
