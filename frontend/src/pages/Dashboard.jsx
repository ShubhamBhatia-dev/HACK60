import React, { useState, useCallback, useEffect } from 'react';
import Sidebar         from '../components/Sidebar';
import EditorTabs      from '../components/EditorTabs';
import MarkdownEditor  from '../components/MarkdownEditor';
import ChatInput       from '../components/ChatInput';
import VersionsSidebar from '../components/VersionsSidebar';
import { useAppStore } from '../store/useAppStore';
import { FileText, History, Cpu } from 'lucide-react';

// Lightweight streaming preview — shown instead of MDXEditor while SSE is active
function StreamingPreview({ content, prompt }) {
  return (
    <div className="streaming-preview">
      <div className="streaming-preview-header">
        <span className="streaming-dot-ring" />
        <Cpu size={13} />
        <span>Generating<span className="streaming-ellipsis" /></span>
        {prompt && <span className="streaming-prompt-label">"{prompt.slice(0, 50)}{prompt.length > 50 ? '…' : ''}"</span>}
      </div>
      <div className="streaming-preview-body">
        <pre className="streaming-text">{content || ' '}</pre>
        <span className="stream-cursor" />
      </div>
    </div>
  );
}

export default function Dashboard() {
  const { openTabs, activeTabId } = useAppStore();
  const activeTab = openTabs.find(t => t.job_id === activeTabId);

  const [showVersions, setShowVersions] = useState(false);
  const [vsWidth, setVsWidth]           = useState(252);

  useEffect(() => {
    if (activeTabId) setShowVersions(true);
  }, [activeTabId]);

  const handleResizeStart = useCallback((e) => {
    e.preventDefault();
    const startX = e.clientX;
    const startW = vsWidth;
    const onMove = (e) => setVsWidth(Math.max(180, Math.min(500, startW + (e.clientX - startX))));
    const onUp   = () => { window.removeEventListener('mousemove', onMove); window.removeEventListener('mouseup', onUp); };
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
  }, [vsWidth]);

  return (
    <div className="dashboard-layout">
      <Sidebar />

      <div className="main-area">
        <div className="tabs-bar-row">
          <EditorTabs />
          {activeTab && (
            <button
              className={`vs-toggle-btn ${showVersions ? 'active' : ''}`}
              onClick={() => setShowVersions(v => !v)}
              title={showVersions ? 'Hide version history' : 'Show version history'}
            >
              <History size={14} />
              Versions
            </button>
          )}
        </div>

        <div className="workspace">
          {showVersions && activeTab && !activeTab.isStreaming && (
            <VersionsSidebar
              tab={activeTab}
              width={vsWidth}
              onResizeStart={handleResizeStart}
            />
          )}

          <div className="workspace-editor">
            {activeTab ? (
              activeTab.isStreaming ? (
                // Show streaming preview while SSE is in progress
                <StreamingPreview
                  content={activeTab.streamingText}
                  prompt={activeTab.prompt}
                />
              ) : (
                <MarkdownEditor key={`${activeTab.job_id}-${activeTab.displayKey ?? 0}`} tab={activeTab} />
              )
            ) : (
              <div className="empty-state">
                <div className="empty-icon"><FileText size={22} /></div>
                <h3>No document open</h3>
                <p>Type a prompt below to generate a Job Description,<br />or pick one from your history.</p>
              </div>
            )}
          </div>
        </div>

        <ChatInput />
      </div>
    </div>
  );
}
