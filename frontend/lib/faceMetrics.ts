// Eye contact, head movement and blink rate, computed from MediaPipe Face Landmarker output.
//
// MediaPipe gives two things per video frame:
//   - 478 face landmarks, each (x, y) normalised to 0..1 across the image
//   - a 4x4 matrix describing how the head is rotated and positioned relative to the camera
// Everything below is geometry on those two outputs. The full explanation is in
// frontend/FACE_METRICS.md.

import { FaceLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";
import type { VisionMetrics } from "./api";

type Point = { x: number; y: number };

// Six landmarks around each eye, in the order the eye-aspect-ratio formula uses:
// [outer corner, upper lid 1, upper lid 2, inner corner, lower lid 2, lower lid 1]
const EYE_A = [33, 160, 158, 133, 153, 144];
const EYE_B = [362, 385, 387, 263, 373, 380];

// Iris centre and the two corners of the same eye, corners listed left-to-right in the image
const IRIS_A = { iris: 468, left: 33, right: 133 };
const IRIS_B = { iris: 473, left: 362, right: 263 };

// A frame counts as eye contact when the head is within these angles of facing the
// camera and the irises sit near the middle of the eyes.
const MAX_HEAD_ANGLE = 15; // degrees, for both yaw and pitch
const MAX_IRIS_OFFSET = 0.15; // distance from the eye's centre, as a share of eye width

// The eye counts as closed below CLOSED and open again above OPEN. Two thresholds
// stop one noisy frame near a single threshold from registering as several blinks.
const EAR_CLOSED = 0.2;
const EAR_OPEN = 0.24;

// Landmarks jitter by a fraction of a degree every frame. Smoothing the head angles
// before measuring movement keeps that jitter from being counted as motion.
const SMOOTHING = 0.3;

function distance(a: Point, b: Point, width: number, height: number) {
  // landmarks are normalised separately in x and y, so convert to pixels first
  return Math.hypot((a.x - b.x) * width, (a.y - b.y) * height);
}

/**
 * Eye aspect ratio: eye height divided by eye width.
 *   EAR = (|p2 - p6| + |p3 - p5|) / (2 * |p1 - p4|)
 * About 0.3 for an open eye, falling toward 0 as it closes. Being a ratio, it does
 * not change with distance from the camera.
 */
export function eyeAspectRatio(landmarks: Point[], eye: number[], width: number, height: number) {
  const [p1, p2, p3, p4, p5, p6] = eye.map((i) => landmarks[i]);
  const vertical = distance(p2, p6, width, height) + distance(p3, p5, width, height);
  return vertical / (2 * distance(p1, p4, width, height));
}

/**
 * Where the iris sits between the eye's two corners: 0 = at the left corner,
 * 0.5 = centred, 1 = at the right corner. The iris is projected onto the
 * corner-to-corner line, so a tilted head does not skew it.
 */
export function irisPosition(landmarks: Point[], eye: typeof IRIS_A, width: number, height: number) {
  const left = landmarks[eye.left];
  const right = landmarks[eye.right];
  const iris = landmarks[eye.iris];
  const ex = (right.x - left.x) * width;
  const ey = (right.y - left.y) * height;
  const ix = (iris.x - left.x) * width;
  const iy = (iris.y - left.y) * height;
  return (ix * ex + iy * ey) / (ex * ex + ey * ey);
}

/**
 * Head yaw (turning left/right) and pitch (nodding up/down) in degrees.
 * The matrix is column-major, so its third column, entries 8..10, is the direction the
 * face points in camera space. Facing the camera that vector is (0, 0, 1) and both
 * angles are 0.
 */
export function headAngles(matrix: ArrayLike<number>) {
  const fx = matrix[8];
  const fy = matrix[9];
  const fz = matrix[10];
  const toDegrees = 180 / Math.PI;
  return {
    yaw: Math.atan2(fx, fz) * toDegrees,
    pitch: Math.atan2(-fy, Math.hypot(fx, fz)) * toDegrees,
  };
}

/** Accumulates per-frame measurements over one answer and reports the totals. */
export class FaceTracker {
  private frames = 0;
  private framesWithFace = 0;
  private framesWithEyeContact = 0;
  private blinks = 0;
  private eyeClosed = false;
  private movementDegrees = 0;
  private trackedSeconds = 0;
  private lastTime: number | null = null;
  private smoothed: { yaw: number; pitch: number } | null = null;

  /** Call when no face was found in a frame. */
  addMissing() {
    this.frames += 1;
    this.lastTime = null;
    this.smoothed = null;
  }

  /** time is in seconds; width and height are the video frame size in pixels. */
  add(time: number, landmarks: Point[], matrix: ArrayLike<number>, width: number, height: number) {
    this.frames += 1;
    this.framesWithFace += 1;

    const { yaw, pitch } = headAngles(matrix);
    const iris =
      (irisPosition(landmarks, IRIS_A, width, height) + irisPosition(landmarks, IRIS_B, width, height)) / 2;
    const facingCamera = Math.abs(yaw) < MAX_HEAD_ANGLE && Math.abs(pitch) < MAX_HEAD_ANGLE;
    if (facingCamera && Math.abs(iris - 0.5) < MAX_IRIS_OFFSET) this.framesWithEyeContact += 1;

    const ear =
      (eyeAspectRatio(landmarks, EYE_A, width, height) + eyeAspectRatio(landmarks, EYE_B, width, height)) / 2;
    if (!this.eyeClosed && ear < EAR_CLOSED) {
      this.eyeClosed = true;
    } else if (this.eyeClosed && ear > EAR_OPEN) {
      this.eyeClosed = false;
      this.blinks += 1; // counted when the eye reopens
    }

    const previous = this.smoothed;
    const smoothed = previous
      ? {
          yaw: previous.yaw + SMOOTHING * (yaw - previous.yaw),
          pitch: previous.pitch + SMOOTHING * (pitch - previous.pitch),
        }
      : { yaw, pitch };
    // skip the gap after a dropped frame or a lost face, where the change is not real motion
    if (previous && this.lastTime !== null && time - this.lastTime < 0.5) {
      this.movementDegrees += Math.hypot(smoothed.yaw - previous.yaw, smoothed.pitch - previous.pitch);
      this.trackedSeconds += time - this.lastTime;
    }
    this.smoothed = smoothed;
    this.lastTime = time;
  }

  result(): VisionMetrics | null {
    if (this.framesWithFace === 0 || this.trackedSeconds === 0) return null;
    return {
      eye_contact_percent: Math.round((100 * this.framesWithEyeContact) / this.framesWithFace),
      // average head rotation speed in degrees per second
      head_movement: Math.round((10 * this.movementDegrees) / this.trackedSeconds) / 10,
      blinks_per_minute: Math.round((10 * this.blinks) / (this.trackedSeconds / 60)) / 10,
      face_visible_percent: Math.round((100 * this.framesWithFace) / this.frames),
      tracked_seconds: Math.round(this.trackedSeconds * 10) / 10,
    };
  }
}

const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.1.0/wasm";
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

let logFilterInstalled = false;

/**
 * MediaPipe's WebAssembly code prints its status lines (such as "INFO: Created TensorFlow
 * Lite XNNPACK delegate for CPU.") through console.error. They are not errors, but the
 * Next.js dev overlay reports every console.error as one, so they are sent to console.info.
 */
function quietMediaPipeInfoLogs() {
  if (logFilterInstalled) return;
  logFilterInstalled = true;
  const original = console.error;
  console.error = (...args: unknown[]) => {
    if (typeof args[0] === "string" && /^\s*(INFO|W\d{4}|I\d{4})\b/.test(args[0])) {
      console.info(...args);
    } else {
      original(...args);
    }
  };
}

/** Downloads the wasm runtime and the model (about 4 MB) and builds a landmarker for video. */
export async function createFaceLandmarker() {
  quietMediaPipeInfoLogs();
  const fileset = await FilesetResolver.forVisionTasks(WASM_URL);
  return FaceLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: MODEL_URL, delegate: "GPU" },
    runningMode: "VIDEO",
    numFaces: 1,
    outputFacialTransformationMatrixes: true,
  });
}
