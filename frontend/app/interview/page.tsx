"use client";

import type { FaceLandmarker } from "@mediapipe/tasks-vision";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useRef, useState } from "react";
import {
  analyzeSpeech,
  endSession,
  nextQuestion,
  submitAnswer,
  type Grade,
  type Question,
  type SpeechMetrics,
  type VisionMetrics,
} from "@/lib/api";
import { createFaceLandmarker, FaceTracker } from "@/lib/faceMetrics";

// loading -> ready -> recording -> processing -> review -> grading -> graded -> (next question) ready
type Phase = "loading" | "ready" | "recording" | "processing" | "review" | "grading" | "graded" | "finishing";

const DIFFICULTY = ["", "Easy", "Medium", "Hard"];

export default function Interview() {
  const router = useRouter();
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const landmarkerRef = useRef<FaceLandmarker | null>(null);
  const trackerRef = useRef<FaceTracker | null>(null);
  const frameRef = useRef(0);

  const [sessionId, setSessionId] = useState<number | null>(null);
  const [phase, setPhase] = useState<Phase>("loading");
  const [question, setQuestion] = useState<Question | null>(null);
  const [answerText, setAnswerText] = useState("");
  const [speech, setSpeech] = useState<SpeechMetrics | null>(null);
  const [vision, setVision] = useState<VisionMetrics | null>(null);
  const [grade, setGrade] = useState<Grade | null>(null);
  const [camera, setCamera] = useState<"starting" | "on" | "off">("starting");
  const [faceTracking, setFaceTracking] = useState<"loading" | "on" | "off">("loading");
  const [seconds, setSeconds] = useState(0);
  const [error, setError] = useState("");

  const finish = useCallback(
    async (id: number) => {
      setPhase("finishing");
      try {
        await endSession(id);
        sessionStorage.removeItem("placeiq.session");
        router.push("/results");
      } catch (e) {
        setError((e as Error).message);
      }
    },
    [router],
  );

  const loadQuestion = useCallback(
    async (id: number) => {
      setPhase("loading");
      setError("");
      try {
        const next = await nextQuestion(id);
        if (next.done || !next.question) {
          await finish(id);
          return;
        }
        setQuestion(next.question);
        setAnswerText("");
        setSpeech(null);
        setVision(null);
        setGrade(null);
        setPhase("ready");
      } catch (e) {
        setError((e as Error).message);
      }
    },
    [finish],
  );

  // pick up the session started on the home page
  useEffect(() => {
    const stored = Number(sessionStorage.getItem("placeiq.session"));
    if (!stored) {
      router.replace("/");
      return;
    }
    setSessionId(stored);
    loadQuestion(stored);
  }, [router, loadQuestion]);

  // webcam and microphone
  useEffect(() => {
    let cancelled = false;
    navigator.mediaDevices
      .getUserMedia({ video: { width: 640, height: 480 }, audio: true })
      .then((stream) => {
        if (cancelled) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }
        streamRef.current = stream;
        if (videoRef.current) videoRef.current.srcObject = stream;
        setCamera("on");
      })
      .catch(() => setCamera("off"));
    return () => {
      cancelled = true;
      cancelAnimationFrame(frameRef.current);
      streamRef.current?.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    };
  }, []);

  // face landmark model
  useEffect(() => {
    let cancelled = false;
    createFaceLandmarker()
      .then((landmarker) => {
        if (cancelled) {
          landmarker.close();
          return;
        }
        landmarkerRef.current = landmarker;
        setFaceTracking("on");
      })
      .catch(() => setFaceTracking("off"));
    return () => {
      cancelled = true;
      landmarkerRef.current?.close();
      landmarkerRef.current = null;
    };
  }, []);

  // recording timer
  useEffect(() => {
    if (phase !== "recording") return;
    setSeconds(0);
    const timer = setInterval(() => setSeconds((s) => s + 1), 1000);
    return () => clearInterval(timer);
  }, [phase]);

  function trackFace() {
    const video = videoRef.current;
    const landmarker = landmarkerRef.current;
    const tracker = trackerRef.current;
    if (video && landmarker && tracker && video.readyState >= 2) {
      const now = performance.now();
      const result = landmarker.detectForVideo(video, now);
      const landmarks = result.faceLandmarks[0];
      const matrix = result.facialTransformationMatrixes[0];
      if (landmarks && matrix) {
        tracker.add(now / 1000, landmarks, matrix.data, video.videoWidth, video.videoHeight);
      } else {
        tracker.addMissing();
      }
    }
    frameRef.current = requestAnimationFrame(trackFace);
  }

  function startRecording() {
    const audioTracks = streamRef.current?.getAudioTracks() ?? [];
    if (audioTracks.length === 0) {
      setError("No microphone available. You can type your answer instead.");
      return;
    }
    setError("");
    const chunks: Blob[] = [];
    // record audio only: the video never leaves the browser
    const recorder = new MediaRecorder(new MediaStream(audioTracks));
    recorder.ondataavailable = (event) => chunks.push(event.data);
    recorder.onstop = () => transcribe(new Blob(chunks, { type: recorder.mimeType }));
    recorder.start();
    recorderRef.current = recorder;

    trackerRef.current = new FaceTracker();
    frameRef.current = requestAnimationFrame(trackFace);
    setPhase("recording");
  }

  function stopRecording() {
    cancelAnimationFrame(frameRef.current);
    setVision(trackerRef.current?.result() ?? null);
    trackerRef.current = null;
    setPhase("processing");
    recorderRef.current?.stop(); // fires onstop -> transcribe
  }

  async function transcribe(audio: Blob) {
    try {
      const result = await analyzeSpeech(audio);
      setAnswerText(result.transcript);
      setSpeech(result.metrics);
    } catch (e) {
      setError(`${(e as Error).message} You can type your answer instead.`);
    }
    setPhase("review");
  }

  async function submit() {
    if (!sessionId || !question) return;
    setPhase("grading");
    setError("");
    try {
      const result = await submitAnswer(sessionId, {
        session_question_id: question.session_question_id,
        answer_text: answerText,
        speech,
        vision,
      });
      setGrade(result);
      setPhase("graded");
    } catch (e) {
      setError((e as Error).message);
      setPhase("review");
    }
  }

  const isLast = question ? question.number >= question.total : false;

  return (
    <div className="interview">
      <section className="card">
        {question ? (
          <>
            <div className="row">
              <span className="muted">
                Question {question.number} of {question.total}
              </span>
              <span className="tag">{question.concept}</span>
              <span className="tag">{DIFFICULTY[question.difficulty]}</span>
              {question.source === "company" && <span className="tag">Company style</span>}
            </div>
            <p className="question">{question.question}</p>
          </>
        ) : (
          <p className="muted">{phase === "finishing" ? "Wrapping up…" : "Loading question…"}</p>
        )}

        {phase === "ready" && (
          <div className="row">
            <button onClick={startRecording} disabled={camera !== "on"}>
              Start answering
            </button>
            <button className="secondary" onClick={() => setPhase("review")}>
              Type instead
            </button>
          </div>
        )}

        {phase === "recording" && (
          <div className="row">
            <span className="recording-dot" aria-hidden />
            <span>
              Recording {Math.floor(seconds / 60)}:{String(seconds % 60).padStart(2, "0")}
            </span>
            <button onClick={stopRecording}>Stop</button>
          </div>
        )}

        {phase === "processing" && <p className="muted">Transcribing your answer…</p>}

        {(phase === "review" || phase === "grading") && (
          <>
            <label htmlFor="answer">Your answer</label>
            <textarea
              id="answer"
              value={answerText}
              onChange={(e) => setAnswerText(e.target.value)}
              placeholder="Type your answer here"
              disabled={phase === "grading"}
            />
            <p className="muted small">This is what will be graded. Fix any words the transcription got wrong.</p>
            <button onClick={submit} disabled={phase === "grading"}>
              {phase === "grading" ? "Grading…" : "Submit answer"}
            </button>
          </>
        )}

        {phase === "graded" && grade && sessionId && (
          <>
            <div className="tiles" style={{ marginBottom: 16 }}>
              <div className="tile">
                <div className="label">Score</div>
                <div className="value">{Math.round(grade.score * 100)}%</div>
                {grade.grader === "keyword" && <div className="note">Keyword match (no LLM grader configured)</div>}
              </div>
              {speech && (
                <div className="tile">
                  <div className="label">Pace</div>
                  <div className="value">{speech.words_per_minute} wpm</div>
                  <div className="note">
                    {speech.filler_count} filler words, {speech.pause_count} long pauses
                  </div>
                </div>
              )}
              {vision && (
                <div className="tile">
                  <div className="label">Eye contact</div>
                  <div className="value">{vision.eye_contact_percent}%</div>
                  <div className="note">{vision.blinks_per_minute} blinks/min</div>
                </div>
              )}
            </div>
            <p>{grade.feedback}</p>
            <details style={{ marginBottom: 16 }}>
              <summary>Reference answer</summary>
              <p style={{ marginTop: 8 }}>{grade.reference_answer}</p>
            </details>
            <button onClick={() => (isLast ? finish(sessionId) : loadQuestion(sessionId))}>
              {isLast ? "Finish and see results" : "Next question"}
            </button>
          </>
        )}

        {error && (
          <p className="error" style={{ marginTop: 14 }}>
            {error}
          </p>
        )}
      </section>

      <aside>
        <video ref={videoRef} className="camera" autoPlay muted playsInline />
        <p className="muted small" style={{ marginTop: 8 }}>
          {camera === "starting" && "Waiting for camera and microphone permission…"}
          {camera === "off" && "Camera and microphone are blocked. You can still type your answers."}
          {camera === "on" && faceTracking === "loading" && "Loading face tracking…"}
          {camera === "on" && faceTracking === "on" && "Face tracking runs in your browser. Only audio is uploaded."}
          {camera === "on" && faceTracking === "off" && "Face tracking could not load. Answers are still recorded."}
        </p>
      </aside>
    </div>
  );
}
