"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import MasteryChart from "@/components/MasteryChart";
import { getProgress, getSession, type Progress, type SessionSummary } from "@/lib/api";

// Plain-language reading of each delivery number. The ranges are common guidance for
// interview speech, not strict rules.
function paceNote(wpm: number) {
  if (wpm < 110) return "On the slow side. Aim for 120-160.";
  if (wpm > 170) return "Quite fast. Aim for 120-160.";
  return "A comfortable pace.";
}

function fillerNote(rate: number) {
  if (rate > 6) return "Frequent. Try pausing silently instead.";
  if (rate > 3) return "Noticeable, but not distracting.";
  return "Very few fillers.";
}

function eyeContactNote(percent: number) {
  if (percent < 50) return "Mostly looking away from the camera.";
  if (percent < 70) return "Fair. Look at the camera when making a point.";
  return "Steady eye contact.";
}

function headNote(degreesPerSecond: number) {
  if (degreesPerSecond > 12) return "A lot of head movement.";
  if (degreesPerSecond < 1.5) return "Very still. A little movement looks natural.";
  return "Natural movement.";
}

function blinkNote(perMinute: number) {
  if (perMinute > 30) return "High. Often a sign of nerves.";
  if (perMinute < 6) return "Low. Possibly staring at the screen.";
  return "In the typical 10-20 range.";
}

function Tile({ label, value, note }: { label: string; value: string; note?: string }) {
  return (
    <div className="tile">
      <div className="label">{label}</div>
      <div className="value">{value}</div>
      {note && <div className="note">{note}</div>}
    </div>
  );
}

export default function Results() {
  const [progress, setProgress] = useState<Progress | null>(null);
  const [latest, setLatest] = useState<SessionSummary | null>(null);
  const [userId, setUserId] = useState<string | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // /results?user=someone shows that user's results; otherwise the last user on this browser
    const user = new URLSearchParams(window.location.search).get("user") ?? localStorage.getItem("placeiq.user");
    setUserId(user);
    if (!user) {
      setLoading(false);
      return;
    }
    getProgress(user)
      .then(async (p) => {
        setProgress(p);
        const last = p.sessions[p.sessions.length - 1];
        if (last) setLatest(await getSession(last.session_id));
      })
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <p className="muted">Loading results…</p>;
  if (error) return <p className="error">{error}</p>;
  if (!userId || !progress || progress.sessions.length === 0) {
    return (
      <>
        <h1>Results</h1>
        <p className="muted">
          No completed interviews yet. <Link href="/">Start one</Link>.
        </p>
      </>
    );
  }

  const speech = latest?.speech;
  const vision = latest?.vision;

  return (
    <>
      <h1>Results for {userId}</h1>
      <p className="muted">
        {progress.sessions.length} session{progress.sessions.length === 1 ? "" : "s"} · focus next on{" "}
        <strong>{progress.weak_concepts.join(", ")}</strong>
      </p>

      <section className="card" style={{ marginTop: 20 }}>
        <h2>Mastery by concept</h2>
        <MasteryChart concepts={progress.concepts} sessions={progress.sessions} weakConcepts={progress.weak_concepts} />
      </section>

      {latest && (
        <>
          <section className="card">
            <h2>Latest session{latest.company ? ` · ${latest.company}` : ""}</h2>
            <div className="tiles">
              <Tile
                label="Average score"
                value={latest.average_score === null ? "–" : `${Math.round(latest.average_score * 100)}%`}
                note={`${latest.answered} question${latest.answered === 1 ? "" : "s"} answered`}
              />
              {speech?.words_per_minute != null && (
                <Tile
                  label="Speaking pace"
                  value={`${Math.round(speech.words_per_minute)} wpm`}
                  note={paceNote(speech.words_per_minute)}
                />
              )}
              {speech?.filler_rate != null && (
                <Tile
                  label="Filler words"
                  value={`${speech.filler_rate.toFixed(1)} per 100`}
                  note={`${speech.filler_count} in total. ${fillerNote(speech.filler_rate)}`}
                />
              )}
              {speech?.longest_pause_seconds != null && (
                <Tile label="Longest pause" value={`${speech.longest_pause_seconds.toFixed(1)} s`} />
              )}
              {vision?.eye_contact_percent != null && (
                <Tile
                  label="Eye contact"
                  value={`${Math.round(vision.eye_contact_percent)}%`}
                  note={eyeContactNote(vision.eye_contact_percent)}
                />
              )}
              {vision?.head_movement != null && (
                <Tile
                  label="Head movement"
                  value={`${vision.head_movement.toFixed(1)}°/s`}
                  note={headNote(vision.head_movement)}
                />
              )}
              {vision?.blinks_per_minute != null && (
                <Tile
                  label="Blink rate"
                  value={`${Math.round(vision.blinks_per_minute)}/min`}
                  note={blinkNote(vision.blinks_per_minute)}
                />
              )}
            </div>
            {!speech && !vision && (
              <p className="muted small" style={{ marginTop: 12 }}>
                Answers in this session were typed, so there is no speech or camera feedback.
              </p>
            )}
          </section>

          <section className="card">
            <h2>Answers</h2>
            {latest.answers.map((answer, i) => (
              <div className="answer" key={i}>
                <div className="row">
                  <strong>{Math.round(answer.score * 100)}%</strong>
                  <span className="tag">{answer.concept}</span>
                  {answer.grader === "keyword" && <span className="tag">Keyword-graded</span>}
                </div>
                <p style={{ margin: "6px 0" }}>{answer.question}</p>
                <p className="muted" style={{ margin: 0 }}>
                  {answer.feedback}
                </p>
              </div>
            ))}
          </section>
        </>
      )}

      <section className="card">
        <h2>All sessions</h2>
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>#</th>
                <th>Date</th>
                <th>Company</th>
                <th>Questions</th>
                <th>Average score</th>
              </tr>
            </thead>
            <tbody>
              {progress.sessions.map((s, i) => (
                <tr key={s.session_id}>
                  <td>{i + 1}</td>
                  <td>{new Date(s.started_at).toLocaleString()}</td>
                  <td>{s.company ?? "General"}</td>
                  <td>{s.answered}</td>
                  <td>{s.average_score === null ? "–" : `${Math.round(s.average_score * 100)}%`}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </>
  );
}
