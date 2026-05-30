"use client";

import { useState } from "react";
import LanguageToggle from "@/components/LanguageToggle";
import PredictionDisplay from "@/components/PredictionDisplay";
import SentenceBuilder from "@/components/SentenceBuilder";
import StatsPanel from "@/components/StatsPanel";
import WebcamStream from "@/components/WebcamStream";
import type { Language, PredictionMessage } from "@/lib/types";

export default function LivePage() {
  const [language, setLanguage] = useState<Language>("asl");
  const [prediction, setPrediction] = useState<PredictionMessage | null>(null);

  return (
    <div className="space-y-6">
      <header className="flex items-center justify-between flex-wrap gap-3">
        <div>
          <h1 className="text-3xl font-bold">Live mode</h1>
          <p className="text-slate-400 text-sm">
            Webcam frames are streamed to the FastAPI backend over WebSocket. Each frame goes
            through the full 8-stage CV pipeline before the result lands here.
          </p>
        </div>
        <LanguageToggle value={language} onChange={setLanguage} />
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <WebcamStream language={language} onPrediction={setPrediction} />
        <div className="space-y-4">
          <PredictionDisplay prediction={prediction} />
          <SentenceBuilder prediction={prediction} language={language} />
          <StatsPanel prediction={prediction} />
        </div>
      </div>
    </div>
  );
}
