"use client";

import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { getCompanies, startSession } from "@/lib/api";

export default function Home() {
  const router = useRouter();
  const [userId, setUserId] = useState("");
  const [company, setCompany] = useState("");
  const [numQuestions, setNumQuestions] = useState(5);
  const [companies, setCompanies] = useState<string[]>([]);
  const [error, setError] = useState("");
  const [starting, setStarting] = useState(false);

  useEffect(() => {
    setUserId(localStorage.getItem("placeiq.user") ?? "");
    // the company list comes from the RAG documents; the form works without it
    getCompanies().then(setCompanies).catch(() => {});
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
      router.push("/interview");
    } catch (e) {
      setError((e as Error).message);
      setStarting(false);
    }
  }

  return (
    <>
      <h1>Practice interview</h1>
      <p className="muted">
        Answer out loud. Each answer is scored, and the next question targets the concepts you are weakest at.
      </p>
      <form className="card" onSubmit={start} style={{ maxWidth: 480, marginTop: 20 }}>
        <div className="field">
          <label htmlFor="user">Your name or ID</label>
          <input id="user" value={userId} onChange={(e) => setUserId(e.target.value)} required maxLength={64} />
          <div className="muted small">Use the same one each time so your progress carries over.</div>
        </div>
        <div className="field">
          <label htmlFor="company">Target company (optional)</label>
          <select id="company" value={company} onChange={(e) => setCompany(e.target.value)}>
            <option value="">General practice</option>
            {companies.map((c) => (
              <option key={c}>{c}</option>
            ))}
          </select>
        </div>
        <div className="field">
          <label htmlFor="count">Number of questions</label>
          <select id="count" value={numQuestions} onChange={(e) => setNumQuestions(Number(e.target.value))}>
            {[3, 5, 8, 10].map((n) => (
              <option key={n}>{n}</option>
            ))}
          </select>
        </div>
        <button disabled={starting || !userId.trim()}>{starting ? "Starting…" : "Start interview"}</button>
        {error && <p className="error" style={{ marginTop: 12 }}>{error}</p>}
      </form>
    </>
  );
}
