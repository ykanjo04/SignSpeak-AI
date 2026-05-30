"use client";

import type { Language } from "@/lib/types";

interface Props {
  value: Language;
  onChange(v: Language): void;
}

const opts: { id: Language; label: string }[] = [
  { id: "asl", label: "ASL" },
  { id: "arsl", label: "ArSL" },
];

export default function LanguageToggle({ value, onChange }: Props) {
  return (
    <div className="card p-3 inline-flex gap-1">
      {opts.map((o) => (
        <button
          key={o.id}
          className={`px-3 py-1.5 rounded-md text-sm font-medium transition ${
            value === o.id
              ? "bg-blue-500/20 text-blue-200 border border-blue-400/40"
              : "text-slate-300 hover:bg-white/5"
          }`}
          onClick={() => onChange(o.id)}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}
