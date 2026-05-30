export type Point2D = { x: number; y: number };
export type Point3D = Point2D & { z: number };

export interface FrameStats {
  fps: number;
  latency_ms: number;
  per_stage_ms: Record<string, number>;
}

export interface PredictionMessage {
  label_id: number;
  label: string;
  display: string;
  confidence: number;
  motion: "stable" | "moving";
  new_letter: boolean;
  edge_density: number;
  hand_landmarks: Point3D[];
  face_landmarks: Point2D[];
  pose_landmarks: Point2D[];
  stats: FrameStats;
  per_model: Record<string, number>;
}

export type Language = "asl" | "arsl";
