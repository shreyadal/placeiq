"use client";

import { useState } from "react";
import type { ProgressPoint } from "@/lib/api";

// One small line chart per concept, all on the same 0-100% scale, weakest concept first.
// Six lines in a single chart would overlap; separate panels let each trend be read
// alone, and the panel title identifies the concept, so no colour legend is needed.

const WIDTH = 240;
const HEIGHT = 120;
const LEFT = 30;
const RIGHT = 40; // room for the end label
const TOP = 12;
const BOTTOM = 20;

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

  return (
    <div className="panel">
      <div className="row" style={{ marginBottom: 4 }}>
        <h3>{concept}</h3>
        {weak && <span className="tag">Focus</span>}
      </div>
      <div style={{ position: "relative" }}>
      <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} role="img" aria-label={`${concept} mastery by session`}>
        {[0, 0.5, 1].map((tick) => (
          <g key={tick}>
            <line
              x1={LEFT}
              x2={WIDTH - RIGHT}
              y1={y(tick)}
              y2={y(tick)}
              stroke={tick === 0 ? "var(--axis)" : "var(--grid)"}
              strokeWidth={1}
            />
            <text x={LEFT - 6} y={y(tick) + 4} textAnchor="end">
              {tick * 100}%
            </text>
          </g>
        ))}
        <text x={x(0)} y={HEIGHT - 4} textAnchor="middle">
          1
        </text>
        {sessions.length > 1 && (
          <text x={x(sessions.length - 1)} y={HEIGHT - 4} textAnchor="middle">
            {sessions.length}
          </text>
        )}

        <path d={path} fill="none" stroke="var(--accent)" strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
        {/* the end point is always marked; other points appear on hover */}
        <circle cx={last.x} cy={last.y} r={4} fill="var(--accent)" stroke="var(--surface)" strokeWidth={2} />
        {hover && (
          <circle
            cx={points[hover.index].x}
            cy={points[hover.index].y}
            r={4}
            fill="var(--accent)"
            stroke="var(--surface)"
            strokeWidth={2}
          />
        )}
        <text className="value" x={last.x + 8} y={last.y + 4}>
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
      <p className="muted small" style={{ margin: "10px 0" }}>
        Horizontal axis: session number. Vertical axis: predicted chance of a good answer on that concept.
      </p>
      <details>
        <summary>Show as table</summary>
        <div className="table-scroll" style={{ marginTop: 8 }}>
          <table>
            <thead>
              <tr>
                <th>Session</th>
                {ordered.map((c) => (
                  <th key={c}>{c}</th>
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
