// Typed calls to the FastAPI backend.

// 127.0.0.1 rather than localhost: on Windows, localhost tries IPv6 first and uvicorn listens on IPv4
export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

export type Question = {
  session_question_id: number;
  question_id: string;
  concept: string;
  difficulty: number;
  question: string;
  source: "bank" | "company";
  number: number;
  total: number;
};

export type SpeechMetrics = {
  word_count: number;
  duration_seconds: number;
  speaking_seconds: number;
  words_per_minute: number;
  filler_count: number;
  fillers: Record<string, number>;
  filler_rate: number;
  pause_count: number;
  longest_pause_seconds: number;
  average_pause_seconds: number;
  total_pause_seconds: number;
};

export type VisionMetrics = {
  eye_contact_percent: number;
  head_movement: number;
  blinks_per_minute: number;
  face_visible_percent: number;
  tracked_seconds: number;
};

export type Grade = {
  score: number;
  feedback: string;
  missing_points: string[];
  grader: "gemini" | "ollama" | "keyword" | "none";
  reference_answer: string;
};

export type GradedAnswer = {
  question: string;
  concept: string;
  difficulty: number;
  answer_text: string;
  score: number;
  feedback: string;
  missing_points: string[];
  grader: string;
  speech: SpeechMetrics | null;
  vision: VisionMetrics | null;
};

export type SessionSummary = {
  session_id: number;
  company: string | null;
  started_at: number;
  answered: number;
  average_score: number | null;
  score_by_concept: Record<string, number>;
  speech: {
    words_per_minute: number | null;
    filler_rate: number | null;
    filler_count: number;
    longest_pause_seconds: number | null;
  } | null;
  vision: {
    eye_contact_percent: number | null;
    head_movement: number | null;
    blinks_per_minute: number | null;
  } | null;
  mastery: Record<string, number>;
  weak_concepts: string[];
  answers: GradedAnswer[];
};

export type ProgressPoint = {
  session_id: number;
  started_at: number;
  company: string | null;
  answered: number;
  average_score: number | null;
  mastery: Record<string, number>;
};

export type Progress = {
  user_id: string;
  concepts: string[];
  sessions: ProgressPoint[];
  mastery: Record<string, number>;
  weak_concepts: string[];
  // "dkt": predicted by the knowledge tracing model; "scores": the user's own average scores
  mastery_source: "dkt" | "scores";
};

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_URL}${path}`, init);
  } catch {
    // a crash inside the backend also lands here: its 500 reply has no CORS headers, so the browser hides it
    throw new Error(`No usable reply from the backend at ${API_URL}. Check that uvicorn is running and look at its window for an error.`);
  }
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(body?.detail ?? `Request failed (${response.status})`);
  }
  return response.json();
}

function post<T>(path: string, body?: unknown): Promise<T> {
  return request<T>(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
}

export type Experience = { text: string; company: string; role: string; year: string; concepts: string };

export const getCompanies = () => request<string[]>("/companies");

export const getExperiences = (company: string, concepts: string[]) =>
  request<Experience[]>(`/companies/${encodeURIComponent(company)}/experiences?concepts=${concepts.join(",")}`);

export const startSession = (user_id: string, company: string | null, num_questions: number) =>
  post<{ session_id: number; num_questions: number }>("/sessions", { user_id, company, num_questions });

export const nextQuestion = (sessionId: number) =>
  request<{ done: boolean; question: Question | null }>(`/sessions/${sessionId}/next`);

export const submitAnswer = (
  sessionId: number,
  body: { session_question_id: number; answer_text: string; speech: SpeechMetrics | null; vision: VisionMetrics | null },
) => post<Grade>(`/sessions/${sessionId}/answers`, body);

export const endSession = (sessionId: number) => post<SessionSummary>(`/sessions/${sessionId}/end`);

export const getSession = (sessionId: number) => request<SessionSummary>(`/sessions/${sessionId}`);

export const getProgress = (userId: string) => request<Progress>(`/users/${encodeURIComponent(userId)}/progress`);

export function analyzeSpeech(audio: Blob) {
  const form = new FormData();
  form.append("audio", audio, "answer.webm");
  return request<{ transcript: string; metrics: SpeechMetrics }>("/speech/analyze", { method: "POST", body: form });
}
