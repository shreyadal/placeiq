"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { ArrowRight, ChevronDown, Mic } from "@/components/Icons";
import { getCompanies, getProgress, startSession, type ProgressPoint } from "@/lib/api";

// answering, transcription, grading and (for a company) question generation, per question
const MINUTES_PER_QUESTION = 2.5;

export default function Home() {
  const router = useRouter();
  const [userId, setUserId] = useState("");
  const [company, setCompany] = useState("");
  const [numQuestions, setNumQuestions] = useState(5);
  const [companies, setCompanies] = useState<string[]>([]);
  const [returning, setReturning] = useState<{ user: string; last: ProgressPoint } | null>(null);
  const [error, setError] = useState("");
  const [starting, setStarting] = useState(false);

  useEffect(() => {
    const stored = localStorage.getItem("placeiq.user") ?? "";
    setUserId(stored);
    // the company list comes from the RAG documents; the form works without it
    getCompanies().then(setCompanies).catch(() => {});
    if (stored) {
      getProgress(stored)
        .then((p) => {
          const last = p.sessions[p.sessions.length - 1];
          if (last) setReturning({ user: stored, last });
        })
        .catch(() => {});
    }
  }, []);

  async function start(event: React.FormEvent) {
    event.preventDefault();
    setStarting(true);
    setError("");
    try {
      const id = userId.trim();
      const session = await startSession(id, company || null, numQuestions);
      localStorage.setItem("placeiq.user", id);
      sessionStorage.setItem("placeiq.session", String(session.session_id));
      sessionStorage.setItem("placeiq.company", company);
      router.push("/interview");
    } catch (e) {
      setError((e as Error).message);
      setStarting(false);
    }
  }

  return (
    <main className="page narrow">
      <div className="stack">
        <h1 className="title">Practice interview</h1>
        <p className="lede">
          Answer out loud. Each answer is scored, and the next question targets the concepts you&apos;re weakest at.
        </p>
      </div>

      <form className="glass form" onSubmit={start}>
        <div className="field-group">
          <label htmlFor="user">Your name or ID</label>
          <input
            id="user"
            className="field"
            value={userId}
            onChange={(e) => setUserId(e.target.value)}
            required
            maxLength={64}
            autoComplete="off"
          />
          <p className="muted small">Use the same one each time so your progress carries over.</p>
        </div>

        <div className="form-grid">
          <div className="field-group">
            <label htmlFor="company">
              Target company <span className="optional">(optional)</span>
            </label>
            <div className="select-wrap">
              <select id="company" className="field" value={company} onChange={(e) => setCompany(e.target.value)}>
                <option value="">General practice</option>
                {companies.map((c) => (
                  <option key={c}>{c}</option>
                ))}
              </select>
              <ChevronDown />
            </div>
          </div>
          <div className="field-group">
            <label htmlFor="count">Number of questions</label>
            <div className="select-wrap">
              <select
                id="count"
                className="field"
                value={numQuestions}
                onChange={(e) => setNumQuestions(Number(e.target.value))}
              >
                {[3, 5, 8, 10].map((n) => (
                  <option key={n}>{n}</option>
                ))}
              </select>
              <ChevronDown />
            </div>
          </div>
        </div>

        <div className="stack">
          <button className="btn primary large block" disabled={starting || !userId.trim()}>
            {starting ? "Starting…" : "Start interview"}
            {!starting && <ArrowRight />}
          </button>
          <p className="icon-note">
            <Mic />
            Needs microphone and camera access · about {Math.round(numQuestions * MINUTES_PER_QUESTION)} minutes
          </p>
          {error && <p className="error">{error}</p>}
        </div>
      </form>

      {returning && (
        <div className="strip">
          <span>
            Welcome back, {returning.user}. Last session:{" "}
            {returning.last.average_score !== null && (
              <>
                <strong>{Math.round(returning.last.average_score * 100)}%</strong> on{" "}
              </>
            )}
            {new Date(returning.last.started_at).toLocaleDateString("en-GB", { day: "numeric", month: "short" })}.
          </span>
          <Link href="/results">View progress</Link>
        </div>
      )}
    </main>
  );
}
