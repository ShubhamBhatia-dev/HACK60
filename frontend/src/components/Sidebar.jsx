import React, { useEffect, useState } from 'react';
import { LogOut, Cpu, Clock, ChevronDown } from 'lucide-react';
import { useAppStore } from '../store/useAppStore';
import { api } from '../api/client';

// Human-readable display names
const MODEL_LABELS = {
  qwen:      'Qwen  (fast)',
  phi:       'Phi-3 (quality)',
  tinyllama: 'TinyLlama (lite)',
};

export default function Sidebar() {
  const { user, logout, history, setHistory, openHistoryItem, selectedModel, setSelectedModel } = useAppStore();
  const [availableModels, setAvailableModels] = useState(['qwen']);

  useEffect(() => {
    api.getHistory().then(data => setHistory(data));
    // Fetch available models from backend
    fetch('http://localhost:8000/models')
      .then(r => r.json())
      .then(d => setAvailableModels(d.models || ['qwen']))
      .catch(() => {});  // silently keep default on error
  }, [setHistory]);

  const initials = user?.name
    ? user.name.split(' ').map(w => w[0]).join('').toUpperCase().slice(0, 2)
    : '?';

  return (
    <div className="sidebar">
      {/* Brand */}
      <div className="sidebar-brand">
        <div className="sidebar-brand-icon">
          <Cpu size={14} />
        </div>
        <span className="sidebar-brand-text">HR SLM</span>
      </div>

      {/* Model selector */}
      <div className="sidebar-model-selector">
        <label className="sidebar-model-label">SLM Model</label>
        <div className="sidebar-model-select-wrap">
          <select
            className="sidebar-model-select"
            value={selectedModel}
            onChange={e => setSelectedModel(e.target.value)}
          >
            {availableModels.map(m => (
              <option key={m} value={m}>
                {MODEL_LABELS[m] || m}
              </option>
            ))}
          </select>
          <ChevronDown size={12} className="sidebar-model-chevron" />
        </div>
      </div>

      {/* History */}
      <div className="sidebar-section-label">History</div>

      <div className="sidebar-content">
        {history.length === 0 ? (
          <div
            style={{
              padding: '1.5rem 0.5rem',
              textAlign: 'center',
              fontSize: '0.75rem',
              color: 'var(--text-3)',
            }}
          >
            No history yet
          </div>
        ) : (
          history.map(item => (
            <div
              key={item.job_id}
              className="history-item"
              onClick={() => openHistoryItem(item)}
            >
              <Clock size={13} className="history-item-icon" />
              <div style={{ minWidth: 0 }}>
                <div className="history-title">{item.prompt}</div>
                <div className="history-date">
                  {new Date(item.timestamp || item.updated_at).toLocaleDateString(undefined, {
                    month: 'short',
                    day: 'numeric',
                  })}
                </div>
              </div>
            </div>
          ))
        )}
      </div>

      {/* Footer */}
      <div className="sidebar-footer">
        <div className="sidebar-user">
          <div className="sidebar-avatar">{initials}</div>
          <span className="sidebar-user-name">{user?.name || 'User'}</span>
        </div>
        <button
          className="btn-icon danger"
          onClick={logout}
          title="Sign out"
        >
          <LogOut size={15} />
        </button>
      </div>
    </div>
  );
}
