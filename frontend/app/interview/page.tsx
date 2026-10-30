"use client";

import type { FaceLandmarker } from "@mediapipe/tasks-vision";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useRef, useState } from "react";
import { ArrowRight, ChevronDown } from "@/components/Icons";
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

function clock(seconds: number) {
  const whole = Math.round(seconds);
  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, "0")}`;
}

export default function Interview() {
  const router = useRouter();
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const landmarkerRef = useRef<FaceLandmarker | null>(null);
  const trackerRef = useRef<FaceTracker | null>(null);
  const frameRef = useRef(0);
  const faceSeenRef = useRef(false);
  const loadRef = useRef(0);

  const [sessionId, setSessionId] = useState<number | null>(null);
  const [company, setCompany] = useState("");
  const [phase, setPhase] = useState<Phase>("loading");
  const [question, setQuestion] = useState<Question | null>(null);
  const [answerText, setAnswerText] = useState("");
  const [speech, setSpeech] = useState<SpeechMetrics | null>(null);
  const [vision, setVision] = useState<VisionMetrics | null>(null);
  const [grade, setGrade] = useState<Grade | null>(null);
  const [camera, setCamera] = useState<"starting" | "on" | "off">("starting");
  const [faceTracking, setFaceTracking] = useState<"loading" | "on" | "off">("loading");
  const [seconds, setSeconds] = useState(0);
  // what the camera sees right now, refreshed once a second while recording
  const [live, setLive] = useState<{ face: boolean; eyeContact: number | null }>({ face: false, eyeContact: null });
  const [error, setError] = useState("");

  const finish = useCallback(
    async (id: number) => {
      setPhase("finishing");
      try {
        await endSession(id);
        sessionStorage.removeItem("placeiq.session");
        sessionStorage.removeItem("placeiq.company");
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
      const request = ++loadRef.current;
      try {
        const next = await nextQuestion(id);
        // a newer request was made meanwhile; a late reply must not replace the question on screen
        if (request !== loadRef.current) return;
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
    setCompany(sessionStorage.getItem("placeiq.company") ?? "");
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

  // recording timer and live camera readout
  useEffect(() => {
    if (phase !== "recording") return;
    setSeconds(0);
    setLive({ face: false, eyeContact: null });
    const timer = setInterval(() => {
      setSeconds((s) => s + 1);
      setLive({
        face: faceSeenRef.current,
        eyeContact: trackerRef.current?.result()?.eye_contact_percent ?? null,
      });
    }, 1000);
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
        faceSeenRef.current = true;
      } else {
        tracker.addMissing();
        faceSeenRef.current = false;
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
    setSpeech(null);
    setVision(null);
    const chunks: Blob[] = [];
    // record audio only: the video never leaves the browser
    const recorder = new MediaRecorder(new MediaStream(audioTracks));
    recorder.ondataavailable = (event) => chunks.push(event.data);
    recorder.onstop = () => transcribe(new Blob(chunks, { type: recorder.mimeType }));
    recorder.start();
    recorderRef.current = recorder;

    faceSeenRef.current = false;
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

  // switching to typing mid-recording throws the recording away
  function typeInstead() {
    if (phase === "recording") {
      cancelAnimationFrame(frameRef.current);
      trackerRef.current = null;
      const recorder = recorderRef.current;
      if (recorder) {
        recorder.onstop = null;
        recorder.stop();
      }
    }
    setError("");
    setPhase("review");
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
  const graded = phase === "graded" && grade !== null;
  const reviewing = phase === "review" || phase === "grading";

  let cameraState = "Camera ready";
  let cameraNote = "";
  if (camera === "starting") cameraState = "Waiting for permission";
  else if (camera === "off") cameraState = "Camera blocked";
  else if (faceTracking === "loading") cameraState = "Loading face tracking";
  else if (faceTracking === "off") cameraState = "Face tracking unavailable";
  else if (phase === "recording") {
    cameraState = live.face ? "Face detected" : "Looking for your face";
    if (live.eyeContact !== null) cameraNote = `Eye contact ${live.eyeContact}%`;
  } else if (graded) {
    cameraState = "Paused";
    cameraNote = "Answer saved";
  }

  return (
    <main className="page">
      {question && (
        <div className="session-bar">
          <span className="meta">
            {company || "General practice"} · {question.total} question{question.total === 1 ? "" : "s"}
          </span>
          <div className="steps" role="img" aria-label={`Progress: question ${question.number} of ${question.total}`}>
            {Array.from({ length: question.total }, (_, i) => (
              <span key={i} className={i < question.number ? "done" : undefined} />
            ))}
          </div>
        </div>
      )}

      <div className="interview">
        <section className={`glass question-card${graded ? " graded" : ""}`}>
          {question ? (
            <>
              <div className="between">
                <span className="meta">
                  Question <strong>{question.number}</strong> of {question.total}
                  {graded && speech && (
                    <>
                      {" "}
                      · <strong>Answered in {clock(speech.duration_seconds)}</strong>
                    </>
                  )}
                </span>
                <div className="pills">
                  <span className="pill">{question.concept}</span>
                  <span className="pill">{DIFFICULTY[question.difficulty]}</span>
                  {question.source === "company" && (
                    <span className="pill accent">{company ? `${company} style` : "Company style"}</span>
                  )}
                </div>
              </div>

              <h1 className="question">{question.question}</h1>
            </>
          ) : (
            <p className="meta">{phase === "finishing" ? "Wrapping up…" : "Loading question…"}</p>
          )}

          {reviewing && (
            <div className="field-group">
              <label htmlFor="answer">Your answer</label>
              <textarea
                id="answer"
                className="area"
                value={answerText}
                onChange={(e) => setAnswerText(e.target.value)}
                placeholder="Type your answer. Typed answers are scored on content only."
                disabled={phase === "grading"}
              />
              {speech && (
                <p className="muted small">This is what will be graded. Fix any words the transcription got wrong.</p>
              )}
            </div>
          )}

          {question && (phase === "loading" || phase === "finishing") && (
            <p role="status" className="meta">
              {phase === "finishing" ? "Wrapping up…" : "Loading the next question…"}
            </p>
          )}

          {phase === "ready" && (
            <div className="between actions">
              <span className="meta">Take a moment, then answer out loud in one to two minutes.</span>
              <div className="buttons">
                <button type="button" className="btn ghost" onClick={typeInstead}>
                  Type instead
                </button>
                <button type="button" className="btn light" onClick={startRecording} disabled={camera !== "on"}>
                  <span className="round" />
                  Start answering
                </button>
              </div>
            </div>
          )}

          {phase === "recording" && (
            <div className="between actions">
              <div role="status" className="status">
                <span className="rec-dot" />
                <span>Recording</span>
                <span className="mono meta">{clock(seconds)}</span>
              </div>
              <div className="buttons">
                <button type="button" className="btn ghost" onClick={typeInstead}>
                  Type instead
                </button>
                <button type="button" className="btn light" onClick={stopRecording}>
                  <span className="square" />
                  Stop
                </button>
              </div>
            </div>
          )}

          {phase === "processing" && (
            <div className="between actions">
              <span role="status" className="meta">
                Transcribing your answer…
              </span>
            </div>
          )}

          {reviewing && (
            <div className="between actions">
              <span role="status" className="meta">
                {phase === "grading"
                  ? "Grading your answer. This can take half a minute."
                  : speech
                    ? "Recording stopped."
                    : "Recording paused while you type."}
              </span>
              <div className="buttons">
                {camera === "on" && (
                  <button type="button" className="btn ghost" onClick={startRecording} disabled={phase === "grading"}>
                    Answer out loud
                  </button>
                )}
                <button
                  type="button"
                  className="btn light"
                  onClick={submit}
                  disabled={phase === "grading" || !answerText.trim()}
                >
                  {phase === "grading" ? "Grading…" : "Submit"}
                </button>
              </div>
            </div>
          )}

          {graded && sessionId && (
            <>
              <div className="tiles">
                <div className="tile">
                  <span className="label">Score</span>
                  <span className="value">
                    {Math.round(grade.score * 100)}
                    <span className="unit">%</span>
                  </span>
                  <div aria-hidden className="bar">
                    <div style={{ width: `${Math.round(grade.score * 100)}%` }} />
                  </div>
                  {grade.grader === "keyword" && <span className="note">Keyword match (approximate)</span>}
                </div>
                {speech && (
                  <div className="tile">
                    <span className="label">Pace</span>
                    <span className="value">
                      {Math.round(speech.words_per_minute)} <span className="unit">wpm</span>
                    </span>
                    <span className="note">
                      {speech.filler_count} filler word{speech.filler_count === 1 ? "" : "s"}, {speech.pause_count} long
                      pause{speech.pause_count === 1 ? "" : "s"}
                    </span>
                  </div>
                )}
                {vision && (
                  <div className="tile">
                    <span className="label">Eye contact</span>
                    <span className="value">
                      {vision.eye_contact_percent}
                      <span className="unit">%</span>
                    </span>
                    <span className="note">{vision.blinks_per_minute} blinks/min</span>
                  </div>
                )}
              </div>

              <div className="field-group">
                <h2 className="label">Feedback</h2>
                <p className="feedback">{grade.feedback}</p>
              </div>

              <details className="reference">
                <summary>
                  Reference answer
                  <ChevronDown size={16} />
                </summary>
                <p>{grade.reference_answer}</p>
              </details>

              <div className="between">
                <span className="meta muted">
                  {isLast ? "That was the last question." : "The next question targets your weakest concepts."}
                </span>
                <button
                  type="button"
                  className="btn primary"
                  onClick={() => (isLast ? finish(sessionId) : loadQuestion(sessionId))}
                >
                  {isLast ? "Finish and see results" : "Next question"}
                  <ArrowRight />
                </button>
              </div>
            </>
          )}

          {error && <p className="error">{error}</p>}
        </section>

        <aside className="camera-col">
          <video ref={videoRef} className="camera" autoPlay muted playsInline />
          <div className="camera-status">
            <span className={`state${phase === "recording" && live.face ? " live" : ""}`}>{cameraState}</span>
            {cameraNote && <span>{cameraNote}</span>}
          </div>
          <p className="muted small">
            {camera === "off"
              ? "Camera and microphone are blocked. You can still type your answers."
              : "Face tracking runs in your browser. Only audio is uploaded."}
          </p>
        </aside>
      </div>
    </main>
  );
}
