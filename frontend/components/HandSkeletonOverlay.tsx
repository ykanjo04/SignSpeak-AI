"use client";

import { useEffect, useRef } from "react";
import type { PredictionMessage } from "@/lib/types";

const HAND_EDGES: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17],
];

interface Props {
  width: number;
  height: number;
  prediction: PredictionMessage | null;
}

export default function HandSkeletonOverlay({ width, height, prediction }: Props) {
  const ref = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const c = ref.current;
    if (!c) return;
    const ctx = c.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, width, height);
    if (!prediction) return;

    if (prediction.face_landmarks.length) {
      ctx.fillStyle = "rgba(96,165,250,0.6)";
      for (const p of prediction.face_landmarks) {
        ctx.fillRect(p.x * width - 1, p.y * height - 1, 2, 2);
      }
    }

    if (prediction.hand_landmarks.length) {
      ctx.strokeStyle = "rgba(251,191,36,0.95)";
      ctx.lineWidth = 2;
      ctx.beginPath();
      for (const [a, b] of HAND_EDGES) {
        const pa = prediction.hand_landmarks[a];
        const pb = prediction.hand_landmarks[b];
        if (!pa || !pb) continue;
        ctx.moveTo(pa.x * width, pa.y * height);
        ctx.lineTo(pb.x * width, pb.y * height);
      }
      ctx.stroke();
      ctx.fillStyle = "rgba(244,114,182,0.95)";
      for (const p of prediction.hand_landmarks) {
        ctx.beginPath();
        ctx.arc(p.x * width, p.y * height, 3.5, 0, 2 * Math.PI);
        ctx.fill();
      }
    }
  }, [prediction, width, height]);

  return (
    <canvas
      ref={ref}
      width={width}
      height={height}
      style={{
        position: "absolute",
        top: 0,
        left: 0,
        width: "100%",
        height: "100%",
        pointerEvents: "none",
      }}
    />
  );
}
