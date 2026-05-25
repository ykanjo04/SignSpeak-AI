"use client";

import { useEffect, useState } from "react";
import type { PredictionMessage } from "@/lib/types";

interface Props {
  prediction: PredictionMessage | null;
}

export default function SentenceBuilder({ prediction }: Props) {
  const [text, setText] = useState<string>("");

  useEffect(() => {
    if (!prediction || !prediction.new_letter) return;
    const next = prediction.display;
    if (!next) return;
    if (prediction.label === "DELETE") {
      setText((t) => t.slice(0, -1));
      return;
    }
    if (prediction.label === "NOTHING") return;
    setText((t) => t + next);
  }, [prediction]);

  const speak = () => {
    if (!text || typeof window === "undefined") return;
    const synth = window.speechSynthesis;
    if (!synth) return;
    const utter = new SpeechSynthesisUtterance(text);
    utter.lang = /[\u0600-\u06FF]/.test(text) ? "ar-SA" : "en-US";
    synth.cancel();
    synth.speak(utter);
  };

  return (
    <div className="card p-6">
      <div className="text-xs uppercase tracking-wider text-slate-400">
        Sentence builder
      </div>
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        rows={3}
        className="mt-2 w-full p-3 rounded-lg bg-black/30 border border-white/10
                   font-mono text-lg outline-none focus:border-blue-400"
        dir="auto"
      />
      <div className="mt-3 flex gap-2">
        <button onClick={speak} disabled={!text} className="btn-primary">
          Speak
        </button>
        <button onClick={() => setText("")} className="btn-secondary">
          Clear
        </button>
        <button
          onClick={() => setText((t) => t + " ")}
          className="btn-secondary"
        >
          Add space
        </button>
      </div>
    </div>
  );
}
