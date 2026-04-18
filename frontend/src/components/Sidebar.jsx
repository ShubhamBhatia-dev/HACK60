import React, { useEffect } from 'react';
import { LogOut, Cpu, Clock } from 'lucide-react';
import { useAppStore } from '../store/useAppStore';
import { api } from '../api/client';

export default function Sidebar() {
  const { user, logout, history, setHistory, openHistoryItem } = useAppStore();

  useEffect(() => {
    api.getHistory().then(data => setHistory(data));
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
              key={item.jobId}
              className="history-item"
              onClick={() => openHistoryItem(item)}
            >
              <Clock size={13} className="history-item-icon" />
              <div style={{ minWidth: 0 }}>
                <div className="history-title">{item.prompt_given_by_user}</div>
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
