import React, { useState } from 'react';
import { Columns2, Clock, Cpu, Bot, GitCompare } from 'lucide-react';
import { useAppStore } from '../store/useAppStore';

// Lightweight inline markdown renderer
function MiniMd({ content }) {
  return (
    <div className="mini-md">
      {(content || '').split('\n').map((line, i) => {
        if (/^### (.+)/.test(line)) return <h3 key={i}>{line.replace(/^### /, '')}</h3>;
        if (/^## (.+)/.test(line))  return <h2 key={i}>{line.replace(/^## /, '')}</h2>;
        if (/^# (.+)/.test(line))   return <h1 key={i}>{line.replace(/^# /, '')}</h1>;
        if (/^[-*] (.+)/.test(line)) return <li key={i}>{line.replace(/^[-*] /, '')}</li>;
        if (line.trim() === '')      return <div key={i} className="mini-md-br" />;
        const parts = line.split(/(\*\*[^*]+\*\*)/g).map((p, j) =>
          /^\*\*/.test(p) ? <strong key={j}>{p.slice(2, -2)}</strong> : p
        );
        return <p key={i}>{parts}</p>;
      })}
    </div>
  );
}

export default function VersionsSidebar({ tab, width, onResizeStart }) {
  const versions    = tab.versions || [];
  const hasBoth     = !!(tab.slm_output && tab.llm_output);
  const setTabDisplay = useAppStore(s => s.setTabDisplay);
  const [view, setView] = useState(hasBoth ? 'compare' : 'history');

  const handleVersionClick = (v, idx) => {
    setTabDisplay(tab.job_id, idx);
  };

  // Restore current (live) content
  const handleRestoreCurrent = () => {
    setTabDisplay(tab.job_id, null);
  };

  return (
    <aside className="vs-panel" style={{ width }}>
      {/* Header */}
      <div className="vs-panel-header">
        <GitCompare size={14} className="vs-panel-icon" />
        <span className="vs-panel-title">Versions</span>
        {versions.length > 0 && (
          <span className="vs-panel-badge">{versions.length}</span>
        )}
      </div>

      {/* Tabs */}
      <div className="vs-panel-tabs">
        {hasBoth && (
          <button
            className={`vs-panel-tab ${view === 'compare' ? 'active' : ''}`}
            onClick={() => setView('compare')}
          >
            <Columns2 size={11} /> Compare
          </button>
        )}
        <button
          className={`vs-panel-tab ${view === 'history' ? 'active' : ''}`}
          onClick={() => setView('history')}
        >
          <Clock size={11} /> History
        </button>
      </div>

      {/* ── Compare (stacked SLM / LLM) ── */}
      {view === 'compare' && hasBoth && (
        <div className="vs-panel-body">
          <div className="vs-block">
            <div className="vs-block-label slm"><Cpu size={11} /> SLM Output</div>
            <div className="vs-block-content"><MiniMd content={tab.slm_output} /></div>
          </div>
          <div className="vs-block-sep" />
          <div className="vs-block llm">
            <div className="vs-block-label llm"><Bot size={11} /> LLM Enhanced</div>
            <div className="vs-block-content"><MiniMd content={tab.llm_output} /></div>
          </div>
        </div>
      )}

      {/* ── History ── */}
      {view === 'history' && (
        <div className="vs-panel-body">
          {versions.length === 0 ? (
            <div className="vs-panel-empty">
              <Clock size={22} style={{ opacity: 0.2 }} />
              <p>No snapshots yet</p>
              <span>Save or Enhance with LLM to create one</span>
            </div>
          ) : (
            <>
              {/* "Current" entry at top — always red (live) */}
              <div
                className={`vs-entry vs-entry--live ${tab.activeVersionIdx == null ? 'vs-entry--active' : ''}`}
                onClick={handleRestoreCurrent}
                title="View current content"
              >
                <div className="vs-entry-top">
                  <span className={`meta-badge ${tab.llm_output ? 'meta-badge--llm' : ''}`}>
                    {tab.llm_output ? 'LLM' : 'SLM'}
                  </span>
                  <span className="vs-entry-label">Current</span>
                  <span className="vs-entry-num">live</span>
                </div>
              </div>

              {/* Saved snapshots, newest first */}
              {[...versions].reverse().map((v, revIdx) => {
                const realIdx = versions.length - 1 - revIdx;
                const isActive = tab.activeVersionIdx === realIdx;
                const sourceClass =
                  v.source === 'LLM'  ? 'vs-entry--llm'  :
                  v.source === 'USER' ? 'vs-entry--user' :
                                       'vs-entry--slm';
                return (
                  <div
                    key={realIdx}
                    className={`vs-entry ${sourceClass} ${isActive ? 'vs-entry--active' : ''}`}
                    onClick={() => handleVersionClick(v, realIdx)}
                    title="Click to view this version in editor"
                  >
                    <div className="vs-entry-top">
                      <span className={`meta-badge ${v.source === 'LLM' ? 'meta-badge--llm' : v.source === 'USER' ? 'meta-badge--user' : ''}`}>
                        {v.source}
                      </span>
                      <span className="vs-entry-label">{v.label}</span>
                      <span className="vs-entry-num">v{realIdx + 1}</span>
                    </div>
                    <div className="vs-entry-time">
                      {new Date(v.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                      {' · '}
                      {new Date(v.timestamp).toLocaleDateString([], { month: 'short', day: 'numeric' })}
                    </div>
                    <p className="vs-entry-preview">
                      {v.content.replace(/[#*\-`]/g, '').trim().slice(0, 90)}…
                    </p>
                  </div>
                );
              })}
            </>
          )}
        </div>
      )}

      {/* Drag-to-resize handle on right edge */}
      <div className="vs-resize-handle" onMouseDown={onResizeStart} title="Drag to resize" />
    </aside>
  );
}
