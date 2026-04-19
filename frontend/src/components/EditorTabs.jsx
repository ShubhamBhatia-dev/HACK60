import React from 'react';
import { X, FileText } from 'lucide-react';
import { useAppStore } from '../store/useAppStore';

export default function EditorTabs() {
  const { openTabs, activeTabId, setActiveTab, closeTab } = useAppStore();

  if (openTabs.length === 0) return null;

  return (
    <div className="tabs-header">
      {openTabs.map(tab => {
        const isActive = activeTabId === tab.job_id;
        const label = tab.prompt
          ? tab.prompt.slice(0, 28) + (tab.prompt.length > 28 ? '…' : '')
          : 'New Job';

        return (
          <div
            key={tab.job_id}
            className={`tab${isActive ? ' active' : ''}`}
            onClick={() => setActiveTab(tab.job_id)}
            title={tab.prompt}
          >
            <FileText size={12} className="tab-icon" />
            <span className="tab-title">{label}</span>
            <button
              className="tab-close"
              onClick={(e) => {
                e.stopPropagation();
                closeTab(tab.job_id);
              }}
              title="Close tab"
            >
              <X size={11} />
            </button>
          </div>
        );
      })}
    </div>
  );
}
