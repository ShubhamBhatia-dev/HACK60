import React, { useState, useRef, useEffect } from 'react';
import { Send, Bot, Cpu, Sparkles, PlusCircle } from 'lucide-react';
import { api } from '../api/client';
import { useAppStore } from '../store/useAppStore';

export default function ChatInput() {
  const [prompt, setPrompt] = useState('');
  const [useLLM, setUseLLM] = useState(false);
  const [loading, setLoading] = useState(false);
  const textareaRef = useRef(null);

  const { addToHistory, activeTabId, openTabs, setActiveTab, selectedModel } = useAppStore();
  const activeTab = openTabs.find(t => t.job_id === activeTabId);

  // mode: 'refine' when a tab is open, 'new' when no active tab
  const mode = activeTab ? 'refine' : 'new';

  // Auto-resize textarea
  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 140) + 'px';
  }, [prompt]);

  const handleNewChat = () => {
    // Deselect active tab → next prompt will create a new tab
    setActiveTab(null);
    setPrompt('');
    textareaRef.current?.focus();
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!prompt.trim() || loading) return;

    const trimmedPrompt = prompt.trim();
    setLoading(true);
    setPrompt('');

    try {
      if (mode === 'refine' && activeTab) {
        // ── Refine mode: stream into the existing tab ──────────────────
        const store = useAppStore.getState();
        const job_id = activeTab.job_id;

        store.setTabStreaming(job_id, '');

        // Pass current best content as context so SLM knows what to refine
        const currentVersions = useAppStore.getState().openTabs.find(t => t.job_id === job_id)?.versions || [];
        const lastContent = currentVersions.length > 0
          ? currentVersions[currentVersions.length - 1].content
          : (activeTab.llm_output || activeTab.slm_output || '');

        await api.streamJobDescription(trimmedPrompt, useLLM, lastContent, selectedModel, {
          onChunk: (text) => useAppStore.getState().setTabStreaming(job_id, text),
          onDone:  (full, source) => {
            const s = useAppStore.getState();
            const isLLM = source === 'LLM';
            s.updateTabContent(job_id, full, isLLM);
            // Auto-snapshot: record this SLM/LLM response in version history
            s.addVersion(job_id, {
              label:     isLLM ? 'LLM Output' : 'SLM Output',
              source:    isLLM ? 'LLM' : 'SLM',
              content:   full,
              timestamp: new Date().toISOString(),
            });
            s.clearTabStreaming(job_id);        // MUST be before setTabDisplay
            s.setTabDisplay(job_id, null);      // bump displayKey → editor remounts
            s.appendTabChatMessage(job_id, { role: 'user',      text: trimmedPrompt, timestamp: new Date().toISOString() });
            s.appendTabChatMessage(job_id, { role: 'assistant', text: full,          timestamp: new Date().toISOString() });
            
            // Update history title/content to reflect latest refinement
            s.updateHistoryItem(job_id, {
              prompt:     trimmedPrompt,
              slm_output: isLLM ? activeTab.slm_output : full,
              llm_output: isLLM ? full : activeTab.llm_output,
              updated_at: new Date().toISOString()
            });

            // Persist to database
            api.saveUserEdits({
              job_id:      job_id,
              prompt:      trimmedPrompt,
              slm_output:  isLLM ? activeTab.slm_output : full,
              llm_output:  isLLM ? full : activeTab.llm_output,
              user_edited: activeTab.user_edited || null
            }).catch(err => console.error('Auto-save failed:', err));
          },
          onError: () => useAppStore.getState().clearTabStreaming(job_id),
        });

      } else {
        // ── New mode: create tab first, then stream into it ────────────
        const tempJobId = 'job_' + Date.now();
        const newItem = {
          job_id:        tempJobId,
          prompt:        trimmedPrompt,
          slm_output:    '',
          llm_output:    null,
          isStreaming:   true,
          streamingText: '',
          timestamp:     new Date().toISOString(),
          chatHistory:   [],
        };

        useAppStore.getState().addToHistory(newItem);
        useAppStore.getState().openHistoryItem(newItem);

        await api.streamJobDescription(trimmedPrompt, useLLM, '', selectedModel, {
          onChunk: (text) => useAppStore.getState().setTabStreaming(tempJobId, text),
          onDone:  (full, source) => {
            const s = useAppStore.getState();
            const isLLM = source === 'LLM';
            s.updateTabContent(tempJobId, full, isLLM);
            // Auto-snapshot: first version entry
            s.addVersion(tempJobId, {
              label:     isLLM ? 'LLM Output' : 'SLM Output',
              source:    isLLM ? 'LLM' : 'SLM',
              content:   full,
              timestamp: new Date().toISOString(),
            });
            s.clearTabStreaming(tempJobId);     // MUST be before setTabDisplay
            s.setTabDisplay(tempJobId, null);   // bump displayKey → editor remounts
            
            // Sync history entry with generated content
            s.updateHistoryItem(tempJobId, {
              slm_output: isLLM ? '' : full,
              llm_output: isLLM ? full : null,
              updated_at: new Date().toISOString()
            });

            // Persist to database
            api.saveUserEdits({
              job_id:      tempJobId,
              prompt:      trimmedPrompt,
              slm_output:  isLLM ? '' : full,
              llm_output:  isLLM ? full : null
            }).catch(err => console.error('Auto-save failed:', err));
          },
          onError: () => useAppStore.getState().clearTabStreaming(tempJobId),
        });
      }
    } catch (error) {
      console.error(error);
      alert('Generation failed: ' + (error.message || error));
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const contextLabel = activeTab
    ? (activeTab.prompt?.slice(0, 44) ?? '') +
    (activeTab.prompt?.length > 44 ? '…' : '')
    : null;

  const placeholderText = mode === 'refine'
    ? `Refine "${contextLabel?.slice(0, 30)}…" — describe what to change`
    : 'Describe the role, e.g. "Senior React engineer, 4 yrs exp, remote"';

  return (
    <div className="chat-container">
      {/* Mode indicator */}
      <div className="chat-mode-bar">
        {mode === 'refine' ? (
          <div className="chat-context-badge">
            <span className="chat-context-badge-dot" />
            <span>Refining: <strong>{contextLabel}</strong></span>
          </div>
        ) : (
          <div className="chat-context-badge chat-context-badge--new">
            <span className="chat-context-badge-dot chat-context-badge-dot--new" />
            <span>New generation</span>
          </div>
        )}

        {/* Switch to New Chat button */}
        {mode === 'refine' && (
          <button
            id="new-chat-btn"
            className="btn-new-chat"
            onClick={handleNewChat}
            title="Start a new generation"
          >
            <PlusCircle size={13} />
            New chat
          </button>
        )}
      </div>

      <form onSubmit={handleSubmit} className="chat-box">
        <div className="chat-input-row">
          <textarea
            id="chat-prompt-input"
            ref={textareaRef}
            className="chat-input"
            rows={1}
            placeholder={placeholderText}
            value={prompt}
            onChange={e => setPrompt(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={loading}
          />
          <button
            id="chat-submit-btn"
            type="submit"
            className="chat-submit"
            disabled={loading || !prompt.trim()}
            title={loading ? 'Generating…' : 'Send'}
          >
            {loading
              ? <Sparkles size={14} className="spin-icon" />
              : <Send size={14} />}
          </button>
        </div>
      </form>
    </div>
  );
}
