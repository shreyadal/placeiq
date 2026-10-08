"use client";

import { useState } from "react";
import type { ProgressPoint } from "@/lib/api";

// One small line chart per concept, all on the same 0-100% scale, weakest concept first.
// Six lines in a single chart would overlap; separate panels let each trend be read
// alone, and the panel title identifies the concept, so no colour legend is needed.

const NAMES: Record<string, string> = {
  DSA: "Data structures and algorithms",
  OS: "Operating systems",
  DBMS: "Databases",
  CN: "Computer networks",
  OOP: "Object-oriented programming",
  ML: "Machine learning",
};

const WIDTH = 320;
const HEIGHT = 140;
const LEFT = 34;
const RIGHT = 58; // room for the end label
const TOP = 12;
const BOTTOM = 28;
// with more sessions than this, only the first and last are labelled on the axis
const MAX_AXIS_LABELS = 6;

type Hover = { index: number; x: number; y: number };

function Panel({ concept, sessions, weak }: { concept: string; sessions: ProgressPoint[]; weak: boolean }) {
  const [hover, setHover] = useState<Hover | null>(null);

  const plotWidth = WIDTH - LEFT - RIGHT;
  const plotHeight = HEIGHT - TOP - BOTTOM;
  const x = (i: number) => LEFT + (sessions.length === 1 ? plotWidth / 2 : (i / (sessions.length - 1)) * plotWidth);
  const y = (p: number) => TOP + (1 - p) * plotHeight;

  const points = sessions.map((s, i) => ({ x: x(i), y: y(s.mastery[concept]), value: s.mastery[concept] }));
  const last = points[points.length - 1];
  const path = points.map((p, i) => `${i === 0 ? "M" : "L"}${p.x},${p.y}`).join(" ");
  const hovered = hover ? sessions[hover.index] : null;
  const labelled = (i: number) => sessions.length <= MAX_AXIS_LABELS || i === 0 || i === sessions.length - 1;
  const values = sessions.map((s) => `${Math.round(s.mastery[concept] * 100)}%`).join(", ");

  return (
    <div className="panel">
      <div className="between">
        <div className="name">
          <h3>{concept}</h3>
          {NAMES[concept] && <span className="muted small">{NAMES[concept]}</span>}
        </div>
        {weak && <span className="pill warn compact">Weak</span>}
      </div>
      <div className="chart">
        <svg
          viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
          role="img"
          aria-label={`${NAMES[concept] ?? concept} mastery across sessions 1 to ${sessions.length}: ${values}`}
        >
          {[0, 0.5, 1].map((tick) => (
            <g key={tick}>
              <line
                x1={LEFT}
                x2={WIDTH - RIGHT}
                y1={y(tick)}
                y2={y(tick)}
                stroke={tick === 0 ? "rgba(255,255,255,0.14)" : "rgba(255,255,255,0.06)"}
                strokeWidth={1}
              />
              <text x={LEFT - 8} y={y(tick) + 4} textAnchor="end">
                {tick * 100}%
              </text>
            </g>
          ))}
          {sessions.map(
            (s, i) =>
              labelled(i) && (
                <text key={s.session_id} x={x(i)} y={HEIGHT - 8} textAnchor="middle">
                  S{i + 1}
                </text>
              ),
          )}

          <path d={path} fill="none" stroke="var(--accent)" strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
          {/* the end point is always marked; other points appear on hover */}
          <circle cx={last.x} cy={last.y} r={3.5} fill="var(--page)" stroke="var(--accent)" strokeWidth={2} />
          {hover && (
            <circle
              cx={points[hover.index].x}
              cy={points[hover.index].y}
              r={3.5}
              fill="var(--accent)"
              stroke="var(--page)"
              strokeWidth={2}
            />
          )}
          <text className="value" x={last.x + 10} y={last.y + 4}>
            {Math.round(last.value * 100)}%
          </text>

          {/* hit targets larger than the marks */}
          {points.map((p, i) => (
            <circle
              key={i}
              cx={p.x}
              cy={p.y}
              r={12}
              fill="transparent"
              onMouseEnter={() => setHover({ index: i, x: p.x, y: p.y })}
              onMouseLeave={() => setHover(null)}
            />
          ))}
        </svg>
        {hover && hovered && (
          <div
            className="tooltip"
            // the svg scales with the panel, so position in percentages of the viewBox
            style={{ left: `${(hover.x / WIDTH) * 100}%`, top: `calc(${(hover.y / HEIGHT) * 100}% - 10px)` }}
          >
            Session {hover.index + 1} · {new Date(hovered.started_at).toLocaleDateString()} ·{" "}
            <strong>{Math.round(hovered.mastery[concept] * 100)}%</strong>
          </div>
        )}
      </div>
    </div>
  );
}

export default function MasteryChart({
  concepts,
  sessions,
  weakConcepts,
}: {
  concepts: string[];
  sessions: ProgressPoint[];
  weakConcepts: string[];
}) {
  const latest = sessions[sessions.length - 1].mastery;
  const ordered = [...concepts].sort((a, b) => latest[a] - latest[b]);

  return (
    <>
      <div className="panels">
        {ordered.map((concept) => (
          <Panel key={concept} concept={concept} sessions={sessions} weak={weakConcepts.includes(concept)} />
        ))}
      </div>
      <details className="table-toggle">
        <summary>Show as table</summary>
        <div className="list table-scroll">
          <table>
            <thead>
              <tr>
                <th scope="col">Session</th>
                {ordered.map((c) => (
                  <th scope="col" key={c}>
                    {c}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {sessions.map((s, i) => (
                <tr key={s.session_id}>
                  <td>
                    {i + 1} · {new Date(s.started_at).toLocaleDateString()}
                  </td>
                  {ordered.map((c) => (
                    <td key={c}>{Math.round(s.mastery[c] * 100)}%</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
    </>
  );
}
