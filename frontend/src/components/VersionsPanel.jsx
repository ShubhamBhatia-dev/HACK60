import React, { useState } from 'react';
import { Columns2, Clock, Cpu, Bot } from 'lucide-react';
import { useAppStore } from '../store/useAppStore';

// Tiny markdown-to-HTML renderer (handles headings, bold, lists, paragraphs)
// No external dep — good enough for read-only preview
function SimpleMarkdown({ content }) {
  const lines = (content || '').split('\n');
  return (
    <div className="simple-md">
      {lines.map((line, i) => {
        if (/^### (.+)/.test(line)) return <h3 key={i}>{line.replace(/^### /, '')}</h3>;
        if (/^## (.+)/.test(line))  return <h2 key={i}>{line.replace(/^## /, '')}</h2>;
        if (/^# (.+)/.test(line))   return <h1 key={i}>{line.replace(/^# /, '')}</h1>;
        if (/^[-*] (.+)/.test(line)) return <li key={i}>{line.replace(/^[-*] /, '')}</li>;
        if (line.trim() === '')      return <br key={i} />;
        // inline bold: **text**
        const parts = line.split(/(\*\*[^*]+\*\*)/g).map((p, j) =>
          /^\*\*/.test(p) ? <strong key={j}>{p.slice(2, -2)}</strong> : p
        );
        return <p key={i}>{parts}</p>;
      })}
    </div>
  );
}

export default function VersionsPanel({ tab }) {
  const hasBoth  = tab.slm_generated && tab.llm_generated;
  const versions = tab.versions || [];

  const [activeView, setActiveView] = useState(hasBoth ? 'compare' : 'history');

  return (
    <div className="versions-panel">
      {/* Sub-nav */}
      <div className="versions-nav">
        {hasBoth && (
          <button
            className={`versions-nav-btn ${activeView === 'compare' ? 'active' : ''}`}
            onClick={() => setActiveView('compare')}
          >
            <Columns2 size={13} />
            Compare SLM vs LLM
          </button>
        )}
        <button
          className={`versions-nav-btn ${activeView === 'history' ? 'active' : ''}`}
          onClick={() => setActiveView('history')}
        >
          <Clock size={13} />
          Version History
          {versions.length > 0 && (
            <span className="versions-count">{versions.length}</span>
          )}
        </button>
      </div>

      {/* ── Compare view ── */}
      {activeView === 'compare' && hasBoth && (
        <div className="compare-grid">
          <VersionColumn label="SLM Output"    source="SLM" content={tab.slm_generated} icon={<Cpu size={12} />} />
          <div className="compare-divider" />
          <VersionColumn label="LLM Enhanced"  source="LLM" content={tab.llm_generated} icon={<Bot size={12} />} highlight />
        </div>
      )}

      {/* ── History view ── */}
      {activeView === 'history' && (
        <div className="history-list">
          {versions.length === 0 ? (
            <div className="versions-empty">
              <p>No saved versions yet. Hit <strong>Save</strong> or <strong>Enhance with LLM</strong> to create snapshots.</p>
            </div>
          ) : (
            versions.map((v, i) => (
              <div key={i} className="version-item">
                <div className="version-item-header">
                  <span className={`meta-badge ${v.source === 'LLM' ? 'meta-badge--llm' : ''}`}>{v.source}</span>
                  <span className="version-label">{v.label}</span>
                  <span className="version-ts">
                    {new Date(v.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                    {' · '}
                    {new Date(v.timestamp).toLocaleDateString([], { month: 'short', day: 'numeric' })}
                  </span>
                </div>
                <div className="version-preview">
                  {v.content.slice(0, 220)}{v.content.length > 220 ? '…' : ''}
                </div>
              </div>
            ))
          )}
        </div>
      )}
    </div>
  );
}

function VersionColumn({ label, source, content, icon, highlight }) {
  return (
    <div className={`version-col ${highlight ? 'version-col--highlight' : ''}`}>
      <div className="version-col-header">
        {icon}
        <span>{label}</span>
        <span className={`meta-badge ${source === 'LLM' ? 'meta-badge--llm' : ''}`}>{source}</span>
      </div>
      <div className="version-col-body">
        <SimpleMarkdown content={content} />
      </div>
    </div>
  );
}
