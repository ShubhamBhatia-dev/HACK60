import React, { useEffect, useState } from 'react';
import { Activity, Brain, Cpu, Clock, TrendingUp, Zap, BarChart2, ChevronDown, ChevronUp } from 'lucide-react';

const BASE = 'http://localhost:8000';

// ── helpers ───────────────────────────────────────────────────────────────────
function fmt(n, decimals = 4) {
  if (n == null || isNaN(n)) return '—';
  return Number(n).toFixed(decimals);
}

function pct(n) {
  if (n == null || isNaN(n)) return '—';
  return (n * 100).toFixed(1) + '%';
}

function dateFmt(iso) {
  if (!iso) return '—';
  return new Date(iso).toLocaleString(undefined, {
    month: 'short', day: 'numeric', year: 'numeric',
    hour: '2-digit', minute: '2-digit',
  });
}

function Badge({ type }) {
  const style = type === 'post_train'
    ? { background: 'var(--accent-dim)', color: 'var(--accent)', border: '1px solid var(--accent)' }
    : { background: 'rgba(34,197,94,.12)', color: '#22c55e', border: '1px solid #22c55e55' };
  return (
    <span style={{
      fontSize: '0.65rem', fontWeight: 700, letterSpacing: '0.06em',
      padding: '2px 8px', borderRadius: 99, ...style,
      textTransform: 'uppercase',
    }}>
      {type === 'post_train' ? 'SFT + DPO' : 'GRPO RL'}
    </span>
  );
}

function Metric({ label, value, sub, accent }) {
  return (
    <div style={{
      background: 'var(--bg-surface)', border: '1px solid var(--border)',
      borderRadius: 'var(--radius)', padding: '1rem 1.2rem',
      display: 'flex', flexDirection: 'column', gap: '0.2rem',
    }}>
      <div style={{ fontSize: '0.68rem', color: 'var(--text-3)', textTransform: 'uppercase', letterSpacing: '0.07em' }}>
        {label}
      </div>
      <div style={{ fontSize: '1.5rem', fontWeight: 700, color: accent || 'var(--text-1)', lineHeight: 1 }}>
        {value}
      </div>
      {sub && <div style={{ fontSize: '0.72rem', color: 'var(--text-3)' }}>{sub}</div>}
    </div>
  );
}

function RunCard({ run, idx }) {
  const [open, setOpen] = useState(idx === 0);
  const m = run.metrics || {};
  const d = run.data    || {};
  const c = run.config  || {};
  const isPost = run.pipeline === 'post_train';

  return (
    <div style={{
      border: '1px solid var(--border)', borderRadius: 'var(--radius)',
      overflow: 'hidden', marginBottom: '1rem',
      background: 'var(--bg-surface)',
    }}>
      {/* Header */}
      <div
        onClick={() => setOpen(o => !o)}
        style={{
          display: 'flex', alignItems: 'center', gap: '0.75rem',
          padding: '0.9rem 1.2rem', cursor: 'pointer',
          background: open ? 'rgba(139,92,246,.05)' : 'transparent',
          transition: 'background 0.2s',
        }}
      >
        <Badge type={run.pipeline} />
        <span style={{ fontWeight: 600, fontSize: '0.88rem', flex: 1 }}>
          Run&nbsp;<code style={{ fontSize: '0.78rem', color: 'var(--text-3)' }}>{run.run_id}</code>
        </span>
       
        <span style={{
          fontSize: '0.72rem', color: 'var(--text-3)',
          background: 'var(--bg-elevated)', border: '1px solid var(--border)',
          padding: '2px 8px', borderRadius: 99,
        }}>
          {run.duration_min ? `${run.duration_min} min` : '—'}
        </span>
        {open ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
      </div>

      {/* Body */}
      {open && (
        <div style={{ padding: '1.2rem', borderTop: '1px solid var(--border)' }}>
          {/* Metric grid */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(140px,1fr))', gap: '0.75rem', marginBottom: '1.2rem' }}>
            {m.perplexity != null && (
              <Metric label="Perplexity" value={fmt(m.perplexity, 2)} sub="lower = better" accent="var(--accent)" />
            )}
            {m.rouge_1 != null && (
              <Metric label="ROUGE-1 F1" value={pct(m.rouge_1)} />
            )}
            {m.rouge_2 != null && (
              <Metric label="ROUGE-2 F1" value={pct(m.rouge_2)} />
            )}
            {m.rouge_l != null && (
              <Metric label="ROUGE-L" value={pct(m.rouge_l)} />
            )}
            {m.avg_reward != null && (
              <Metric label="Avg Reward" value={fmt(m.avg_reward)} sub={`${fmt(m.min_reward)} – ${fmt(m.max_reward)}`} accent="#22c55e" />
            )}
          </div>

          {/* Config + data row */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
            <div>
              <div style={{ fontSize: '0.68rem', color: 'var(--text-3)', textTransform: 'uppercase', letterSpacing: '.07em', marginBottom: '.5rem' }}>
                Config
              </div>
              <table style={{ fontSize: '0.78rem', width: '100%', borderCollapse: 'collapse' }}>
                <tbody>
                  {Object.entries(c).map(([k, v]) => (
                    <tr key={k}>
                      <td style={{ color: 'var(--text-3)', padding: '2px 0', paddingRight: '1rem' }}>{k}</td>
                      <td style={{ color: 'var(--text-1)', fontWeight: 500 }}>{String(v)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div>
              <div style={{ fontSize: '0.68rem', color: 'var(--text-3)', textTransform: 'uppercase', letterSpacing: '.07em', marginBottom: '.5rem' }}>
                Dataset
              </div>
              <table style={{ fontSize: '0.78rem', width: '100%', borderCollapse: 'collapse' }}>
                <tbody>
                  {Object.entries(d).map(([k, v]) => (
                    <tr key={k}>
                      <td style={{ color: 'var(--text-3)', padding: '2px 0', paddingRight: '1rem' }}>{k}</td>
                      <td style={{ color: 'var(--text-1)', fontWeight: 500 }}>{String(v)}</td>
                    </tr>
                  ))}
                  <tr>
                    <td style={{ color: 'var(--text-3)', padding: '2px 0', paddingRight: '1rem' }}>base model</td>
                    <td style={{ color: 'var(--text-1)', fontWeight: 500, fontSize: '0.72rem' }}>{run.model_base}</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ── Trend spark line ──────────────────────────────────────────────────────────
function SparkLine({ values, color = 'var(--accent)', label }) {
  const w = 200, h = 50, pad = 4;
  if (!values || values.length < 2) return null;
  const min = Math.min(...values), max = Math.max(...values);
  const range = max - min || 1;
  const pts = values.map((v, i) => {
    const x = pad + (i / (values.length - 1)) * (w - pad * 2);
    const y = h - pad - ((v - min) / range) * (h - pad * 2);
    return `${x},${y}`;
  }).join(' ');

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.3rem' }}>
      <div style={{ fontSize: '0.68rem', color: 'var(--text-3)', textTransform: 'uppercase', letterSpacing: '.07em' }}>{label}</div>
      <svg width={w} height={h}>
        <polyline points={pts} fill="none" stroke={color} strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
        {values.map((v, i) => {
          const x = pad + (i / (values.length - 1)) * (w - pad * 2);
          const y = h - pad - ((v - min) / range) * (h - pad * 2);
          return <circle key={i} cx={x} cy={y} r={3} fill={color} />;
        })}
      </svg>
      <div style={{ fontSize: '0.75rem', color: 'var(--text-1)', fontWeight: 600 }}>
        Latest: {values[values.length - 1]?.toFixed(4)}
      </div>
    </div>
  );
}

// ── Page ──────────────────────────────────────────────────────────────────────
export default function AccuracyPage() {
  const [runs, setRuns] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch(`${BASE}/training-runs?limit=20`)
      .then(r => { if (!r.ok) throw new Error('Failed to load'); return r.json(); })
      .then(data => { setRuns(data); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  const latest = runs[0];
  const perplexities = runs.map(r => r.metrics?.perplexity).filter(Boolean).reverse();
  const rouge1s      = runs.map(r => r.metrics?.rouge_1).filter(Boolean).reverse();
  const rewards      = runs.map(r => r.metrics?.avg_reward).filter(Boolean).reverse();

  return (
    <div style={{ minHeight: '100vh', background: 'var(--bg-base)', color: 'var(--text-1)', fontFamily: 'var(--font)' }}>
      {/* Header */}
      <div style={{
        borderBottom: '1px solid var(--border)',
        padding: '1.5rem 2rem',
        display: 'flex', alignItems: 'center', gap: '1rem',
      }}>
        <div style={{
          width: 36, height: 36, borderRadius: 8,
          background: 'var(--accent-dim)', display: 'flex', alignItems: 'center', justifyContent: 'center',
        }}>
          <Activity size={18} color="var(--accent)" />
        </div>
        <div>
          <h1 style={{ margin: 0, fontSize: '1.2rem', fontWeight: 700 }}>Model Accuracy & Training Metrics</h1>
          <p style={{ margin: 0, fontSize: '0.78rem', color: 'var(--text-3)' }}>
            Live view of all training runs — GRPO Reinforcement Learning &amp; SFT+DPO Post-Training
          </p>
        </div>
      </div>

      <div style={{ padding: '2rem', maxWidth: 1100, margin: '0 auto' }}>
        {loading && (
          <div style={{ textAlign: 'center', padding: '4rem', color: 'var(--text-3)' }}>
            <Zap size={24} style={{ opacity: 0.4, marginBottom: '1rem' }} />
            <p>Loading training runs…</p>
          </div>
        )}

        {error && (
          <div style={{
            background: 'rgba(239,68,68,.1)', border: '1px solid rgba(239,68,68,.3)',
            borderRadius: 'var(--radius)', padding: '1rem 1.5rem', color: '#f87171',
            marginBottom: '1.5rem',
          }}>
            ⚠ {error}
          </div>
        )}

        {!loading && !error && runs.length === 0 && (
          <div style={{ textAlign: 'center', padding: '4rem', color: 'var(--text-3)' }}>
            <Brain size={32} style={{ opacity: 0.2, marginBottom: '1rem' }} />
            <p>No training runs yet. Runs appear here automatically after the pipeline completes.</p>
          </div>
        )}

        {!loading && runs.length > 0 && (
          <>
            {/* Summary cards */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(160px,1fr))', gap: '1rem', marginBottom: '2rem' }}>
              <Metric label="Total Runs"       value={runs.length}                               sub="all pipelines"        accent="var(--accent)" />
              <Metric label="Latest Perplexity" value={fmt(latest?.metrics?.perplexity, 2)}     sub="lower = better"       accent="var(--accent)" />
              <Metric label="Latest ROUGE-1"    value={pct(latest?.metrics?.rouge_1)}            sub="token F1"             />
              <Metric label="Latest ROUGE-L"    value={pct(latest?.metrics?.rouge_l)}            sub="LCS F1"               />
              <Metric label="Latest Avg Reward" value={fmt(latest?.metrics?.avg_reward)}         sub="GRPO runs only"       accent="#22c55e" />
              <Metric label="Last Trained"      value={dateFmt(latest?.started_at)?.split(',')[0]} sub={latest?.pipeline}  />
            </div>

            {/* Trend lines */}
            {perplexities.length >= 2 && (
              <div style={{
                display: 'flex', gap: '3rem', flexWrap: 'wrap',
                background: 'var(--bg-surface)', border: '1px solid var(--border)',
                borderRadius: 'var(--radius)', padding: '1.5rem',
                marginBottom: '2rem',
              }}>
                <SparkLine values={perplexities} color="var(--accent)"  label="Perplexity over runs" />
                {rouge1s.length >= 2 && <SparkLine values={rouge1s} color="#22c55e" label="ROUGE-1 over runs" />}
                {rewards.length >= 2 && <SparkLine values={rewards}  color="#f59e0b" label="Avg Reward over runs" />}
              </div>
            )}

            {/* Run list */}
            <div style={{ marginBottom: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <BarChart2 size={14} color="var(--text-3)" />
              <span style={{ fontSize: '0.78rem', color: 'var(--text-3)', textTransform: 'uppercase', letterSpacing: '.07em', fontWeight: 600 }}>
                Training Run History
              </span>
            </div>
            {runs.map((run, i) => <RunCard key={run.run_id} run={run} idx={i} />)}
          </>
        )}
      </div>
    </div>
  );
}
