import React from 'react';
import { X, FileText } from 'lucide-react';
import { useAppStore } from '../store/useAppStore';

export default function EditorTabs() {
  const { openTabs, activeTabId, setActiveTab, closeTab } = useAppStore();

  if (openTabs.length === 0) return null;

  return (
    <div className="tabs-header">
      {openTabs.map(tab => {
        const isActive = activeTabId === tab.jobId;
        const label = tab.prompt_given_by_user
          ? tab.prompt_given_by_user.slice(0, 28) + (tab.prompt_given_by_user.length > 28 ? '…' : '')
          : 'New Job';

        return (
          <div
            key={tab.jobId}
            className={`tab${isActive ? ' active' : ''}`}
            onClick={() => setActiveTab(tab.jobId)}
            title={tab.prompt_given_by_user}
          >
            <FileText size={12} className="tab-icon" />
            <span className="tab-title">{label}</span>
            <button
              className="tab-close"
              onClick={(e) => {
                e.stopPropagation();
                closeTab(tab.jobId);
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
