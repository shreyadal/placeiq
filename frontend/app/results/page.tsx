"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import MasteryChart from "@/components/MasteryChart";
import {
  getExperiences,
  getProgress,
  getSession,
  type Experience,
  type Progress,
  type SessionSummary,
} from "@/lib/api";

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

function shortDate(ms: number) {
  return new Date(ms).toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" });
}

function plural(count: number, word: string) {
  return `${count} ${word}${count === 1 ? "" : "s"}`;
}

function Tile({ label, value, unit, note }: { label: string; value: string; unit?: string; note?: string }) {
  return (
    <div className="tile">
      <span className="label">{label}</span>
      <span className="value">
        {value} {unit && <span className="unit">{unit}</span>}
      </span>
      {note && <span className="note">{note}</span>}
    </div>
  );
}

export default function Results() {
  const [progress, setProgress] = useState<Progress | null>(null);
  const [latest, setLatest] = useState<SessionSummary | null>(null);
  const [userId, setUserId] = useState<string | null>(null);
  const [company, setCompany] = useState<string | null>(null);
  const [experiences, setExperiences] = useState<Experience[]>([]);
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
        // the most recent session that targeted a company decides which interview notes to show
        const company = [...p.sessions].reverse().find((s) => s.company)?.company;
        if (company) {
          setCompany(company);
          getExperiences(company, p.weak_concepts).then(setExperiences).catch(() => {});
        }
      })
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <main className="page results">
        <p role="status" className="meta">
          Loading results…
        </p>
      </main>
    );
  }
  if (error) {
    return (
      <main className="page results">
        <p className="error">{error}</p>
      </main>
    );
  }
  if (!userId || !progress || progress.sessions.length === 0) {
    return (
      <main className="page results">
        <div className="between results-head">
          <div>
            <h1 className="title">Results</h1>
            <p className="lede">No completed interviews yet.</p>
          </div>
          <Link href="/" className="btn primary">
            Start an interview
          </Link>
        </div>
      </main>
    );
  }

  const speech = latest?.speech;
  const vision = latest?.vision;
  const count = progress.sessions.length;

  return (
    <main className="page results">
      <div className="between results-head">
        <div>
          <h1 className="title">Results for {userId}</h1>
          <p className="lede">
            {plural(count, "session")} · focus next on <strong>{progress.weak_concepts.join(", ")}</strong>
          </p>
        </div>
      </div>

      <section className="section">
        <div className="between baseline">
          <h2>Mastery by concept</h2>
          <span className="muted small">
            {count === 1 ? "Session 1" : `Sessions 1–${count}`} · weakest first ·{" "}
            {progress.mastery_source === "dkt" ? "predicted chance of a good answer" : "average score so far"}
          </span>
        </div>
        <MasteryChart concepts={progress.concepts} sessions={progress.sessions} weakConcepts={progress.weak_concepts} />
      </section>

      {latest && (
        <>
          <section className="section">
            <div className="between baseline">
              <h2>Latest session</h2>
              <span className="muted small">
                {shortDate(latest.started_at)} · {latest.company ?? "General practice"} ·{" "}
                {plural(latest.answered, "question")}
              </span>
            </div>
            <div className="tiles wide">
              <Tile
                label="Average score"
                value={latest.average_score === null ? "–" : String(Math.round(latest.average_score * 100))}
                unit={latest.average_score === null ? undefined : "%"}
                note={`${plural(latest.answered, "question")} answered.`}
              />
              {speech?.words_per_minute != null && (
                <Tile
                  label="Speaking pace"
                  value={String(Math.round(speech.words_per_minute))}
                  unit="wpm"
                  note={paceNote(speech.words_per_minute)}
                />
              )}
              {speech?.filler_rate != null && (
                <Tile
                  label="Filler words"
                  value={speech.filler_rate.toFixed(1)}
                  unit="per 100"
                  note={`${speech.filler_count} in total. ${fillerNote(speech.filler_rate)}`}
                />
              )}
              {speech?.longest_pause_seconds != null && (
                <Tile label="Longest pause" value={speech.longest_pause_seconds.toFixed(1)} unit="s" />
              )}
              {vision?.eye_contact_percent != null && (
                <Tile
                  label="Eye contact"
                  value={String(Math.round(vision.eye_contact_percent))}
                  unit="%"
                  note={eyeContactNote(vision.eye_contact_percent)}
                />
              )}
              {vision?.head_movement != null && (
                <Tile
                  label="Head movement"
                  value={vision.head_movement.toFixed(1)}
                  unit="°/s"
                  note={headNote(vision.head_movement)}
                />
              )}
              {vision?.blinks_per_minute != null && (
                <Tile
                  label="Blink rate"
                  value={String(Math.round(vision.blinks_per_minute))}
                  unit="/min"
                  note={blinkNote(vision.blinks_per_minute)}
                />
              )}
            </div>
            {!speech && !vision && (
              <p className="muted small">
                Answers in this session were typed, so there is no speech or camera feedback.
              </p>
            )}
          </section>

          <section className="section">
            <h2>Answers</h2>
            <div className="list">
              {latest.answers.map((answer, i) => (
                <div className="list-row" key={i}>
                  <span className="score">
                    {Math.round(answer.score * 100)}
                    <span>%</span>
                  </span>
                  <div className="body">
                    <div className="head">
                      <span className="pill compact">{answer.concept}</span>
                      {answer.grader === "keyword" && <span className="pill compact">Keyword-graded</span>}
                      <span>{answer.question}</span>
                    </div>
                    <p>{answer.feedback}</p>
                  </div>
                </div>
              ))}
            </div>
          </section>
        </>
      )}

      {company && experiences.length > 0 && (
        <section className="section">
          <div className="between baseline">
            <h2>What {company} has asked</h2>
            <span className="muted small">Interview notes on {progress.weak_concepts.join(", ")}</span>
          </div>
          <div className="list">
            {experiences.map((e, i) => (
              <div className="list-row" key={i}>
                <div className="body">
                  <div className="head">
                    <span className="pill compact">{e.concepts}</span>
                    <span>{e.role}</span>
                  </div>
                  <p>{e.text}</p>
                </div>
              </div>
            ))}
          </div>
        </section>
      )}

      <section className="section">
        <h2>All sessions</h2>
        <div className="list table-scroll">
          <table>
            <thead>
              <tr>
                <th scope="col">#</th>
                <th scope="col">Date</th>
                <th scope="col">Company</th>
                <th scope="col">Questions</th>
                <th scope="col" className="num">
                  Average score
                </th>
              </tr>
            </thead>
            <tbody>
              {/* newest first */}
              {progress.sessions
                .map((s, i) => ({ ...s, number: i + 1 }))
                .reverse()
                .map((s) => (
                  <tr key={s.session_id}>
                    <td className="index">{s.number}</td>
                    <td>{shortDate(s.started_at)}</td>
                    <td>{s.company ?? "General practice"}</td>
                    <td>{s.answered}</td>
                    <td className="num">{s.average_score === null ? "–" : `${Math.round(s.average_score * 100)}%`}</td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      </section>
    </main>
  );
}
