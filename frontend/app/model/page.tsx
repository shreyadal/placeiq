"use client";

import { useEffect, useState } from "react";
import { ChevronDown } from "@/components/Icons";
import {
  checkCandidate,
  getModelCandidates,
  getModelMetrics,
  type CandidateCheck,
  type DatasetMetrics,
  type ModelMetrics,
} from "@/lib/api";

// Every number on this page comes from GET /model/metrics (ml/artifacts/metrics.json, written
// by ml.train and ml.test_model) or from GET /model/check, which runs the model live.

const DATASET_NAMES: Record<string, string> = {
  interview: "Simulated interview data",
  assist2009: "ASSISTments 2009",
};

function percent(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}

function duration(seconds: number) {
  if (seconds < 60) return `${seconds.toFixed(1)} s`;
  return `${Math.floor(seconds / 60)} min ${Math.round(seconds % 60)} s`;
}

function count(n: number) {
  return n.toLocaleString("en-IN");
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

function DatasetSection({ m }: { m: DatasetMetrics }) {
  return (
    <section className="section">
      <div className="between baseline">
        <h2>{DATASET_NAMES[m.dataset] ?? m.dataset}</h2>
        <span className="muted small">predicting the next answer of held-out students</span>
      </div>
      <div className="tiles wide five">
        <Tile label="Test AUC" value={m.test_auc.toFixed(4)} note="0.5 is chance, 1 is perfect." />
        <Tile
          label="Leak-free test AUC"
          value={m.test_auc_no_repeat.toFixed(4)}
          note={
            m.test_auc_no_repeat === m.test_auc
              ? "Same as above: every question has one concept, so there are no repeat rows."
              : "Without the repeat rows of multi-skill questions, whose answer the model has already seen."
          }
        />
        <Tile label="Test accuracy" value={percent(m.test_acc)} note={`Validation AUC ${m.valid_auc.toFixed(4)}.`} />
        <Tile
          label="Best epoch"
          value={String(m.best_epoch)}
          note={`Stopped after ${m.epochs_run} of max ${m.max_epochs} epochs (early stopping).`}
        />
        <Tile label="Training time" value={duration(m.train_seconds)} note="Laptop CPU." />
      </div>
      <div className="list table-scroll">
        <table>
          <thead>
            <tr>
              <th scope="col">Split</th>
              <th scope="col" className="num">
                Students
              </th>
              <th scope="col" className="num">
                Answers
              </th>
            </tr>
          </thead>
          <tbody>
            {(
              [
                ["Train (folds 1–4)", m.train_students, m.train_answers],
                ["Validation (fold 0)", m.valid_students, m.valid_answers],
                ["Test (held out)", m.test_students, m.test_answers],
              ] as const
            ).map(([name, students, answers]) => (
              <tr key={name}>
                <td>{name}</td>
                <td className="num">{count(students)}</td>
                <td className="num">{count(answers)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="muted small mono">{m.command}</p>
    </section>
  );
}

function CandidatePanel({ candidates }: { candidates: string[] }) {
  const [selected, setSelected] = useState(candidates[0]);
  const [check, setCheck] = useState<CandidateCheck | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    setError("");
    checkCandidate(selected)
      .then(setCheck)
      .catch((e: Error) => setError(e.message));
  }, [selected]);

  // weakest first by true ability
  const concepts = check ? Object.keys(check.true_ability).sort((a, b) => check.true_ability[a] - check.true_ability[b]) : [];

  return (
    <div className="panel check-panel">
      <div className="between">
        <h3>Check on a simulated candidate</h3>
        <div className="candidate-pick">
          <label htmlFor="candidate">Candidate</label>
          <div className="select-wrap">
            <select id="candidate" className="field" value={selected} onChange={(e) => setSelected(e.target.value)}>
              {candidates.map((c) => (
                <option key={c}>{c}</option>
              ))}
            </select>
            <ChevronDown />
          </div>
        </div>
      </div>
      {error && <p className="error">{error}</p>}
      {check && (
        <>
          <div className="legend small">
            <span>
              <i className="swatch predicted" /> Predicted by DKT
            </span>
            <span>
              <i className="swatch truth" /> True (simulator)
            </span>
          </div>
          <div className="pairs" role="list">
            {concepts.map((c) => (
              <div className="pair" role="listitem" key={c}>
                <span className="concept">
                  {c}
                  {c === check.true_weakest && <span className="pill warn compact">Weakest</span>}
                </span>
                <div className="bars">
                  <div className="bar-row">
                    <div className="track">
                      <div className="fill predicted" style={{ width: percent(check.predicted_mastery[c]) }} />
                    </div>
                    <span className="num">{percent(check.predicted_mastery[c])}</span>
                  </div>
                  <div className="bar-row">
                    <div className="track">
                      <div className="fill truth" style={{ width: percent(check.true_p_good_medium[c]) }} />
                    </div>
                    <span className="num">{percent(check.true_p_good_medium[c])}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
          <p className="muted small">
            {check.answers} answers in this candidate&apos;s history. Spearman{" "}
            <strong>{check.spearman === null ? "–" : check.spearman.toFixed(2)}</strong> · true weakest{" "}
            <strong>{check.true_weakest}</strong> {check.true_weakest_in_predicted_2 ? "is" : "is not"} in the predicted 2
            weakest ({check.predicted_weakest_2.join(", ")}). Predicted: chance of a good answer. True: the simulator&apos;s
            chance of a good answer on a medium question. Only the ranking is scored.
          </p>
        </>
      )}
    </div>
  );
}

export default function ModelPage() {
  const [metrics, setMetrics] = useState<ModelMetrics | null>(null);
  const [candidates, setCandidates] = useState<string[]>([]);
  const [error, setError] = useState("");

  useEffect(() => {
    getModelMetrics()
      .then(setMetrics)
      .catch((e: Error) => setError(e.message));
    getModelCandidates()
      .then(setCandidates)
      .catch(() => {});
  }, []);

  const weak = metrics?.weak_concepts;

  return (
    <main className="page results">
      <div className="results-head">
        <div>
          <h1 className="title">Knowledge tracing model</h1>
          <p className="lede">
            A DKT model reads a candidate&apos;s answer history and predicts their chance of a good answer on each concept.
            The interview picks the next concept from these predictions.
          </p>
        </div>
      </div>

      <p className="strip">
        <span>
          <strong>Note:</strong> trained on ASSISTments 2009 and simulated interview data. Not yet validated on real
          students. First un-tuned run.
        </span>
      </p>

      {error && <p className="error">{error}</p>}
      {!metrics && !error && (
        <p role="status" className="meta">
          Loading metrics…
        </p>
      )}

      {metrics?.interview && <DatasetSection m={metrics.interview} />}
      {metrics?.assist2009 && <DatasetSection m={metrics.assist2009} />}

      {weak && (
        <section className="section">
          <div className="between baseline">
            <h2>Does it find each candidate&apos;s weak concepts?</h2>
            <span className="muted small">{weak.candidates} held-out simulated candidates</span>
          </div>
          <div className="tiles wide">
            <Tile
              label="Mean Spearman, DKT"
              value={weak.dkt.mean_spearman.toFixed(3)}
              note={`Predicted vs true ranking of the 6 concepts. Median ${weak.dkt.median_spearman.toFixed(3)}. 0 is chance.`}
            />
            <Tile
              label="True weakest in predicted 2, DKT"
              value={percent(weak.dkt.weakest_in_top2_rate)}
              note={`${weak.dkt.weakest_in_top2} of ${weak.candidates}. Random guessing: ${percent(weak.random_top2_rate)}.`}
            />
            <Tile
              label="Baseline: average score"
              value={weak.score_average.mean_spearman.toFixed(3)}
              note={`Mean Spearman; true weakest in predicted 2 for ${percent(weak.score_average.weakest_in_top2_rate)}. The simple average currently does better than this DKT run.`}
            />
          </div>
          {candidates.length > 0 && <CandidatePanel candidates={candidates} />}
          <p className="muted small mono">{weak.command}</p>
        </section>
      )}
    </main>
  );
}
