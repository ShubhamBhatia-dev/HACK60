import React, { useState, useRef, useEffect } from 'react';
import { Send, Bot, Cpu, Sparkles, PlusCircle } from 'lucide-react';
import { api } from '../api/client';
import { useAppStore } from '../store/useAppStore';

export default function ChatInput() {
  const [prompt, setPrompt] = useState('');
  const [useLLM, setUseLLM] = useState(false);
  const [loading, setLoading] = useState(false);
  const textareaRef = useRef(null);

  const { addToHistory, activeTabId, openTabs, setActiveTab } = useAppStore();
  const activeTab = openTabs.find(t => t.jobId === activeTabId);

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
        const { setTabStreaming, clearTabStreaming, updateTabContent, setTabDisplay, appendTabChatMessage } = useAppStore.getState();
        const jobId = activeTab.jobId;

        // Mark tab as streaming immediately
        setTabStreaming(jobId, '');

        await api.streamJobDescription(trimmedPrompt, useLLM, {
          onChunk: (text) => setTabStreaming(jobId, text),
          onDone:  (full, source) => {
            const isLLM = source === 'LLM';
            updateTabContent(jobId, full, isLLM);
            setTabDisplay(jobId, full, null);
            clearTabStreaming(jobId);
            appendTabChatMessage(jobId, { role: 'user',      text: trimmedPrompt,  timestamp: new Date().toISOString() });
            appendTabChatMessage(jobId, { role: 'assistant', text: full,           timestamp: new Date().toISOString() });
          },
          onError: () => clearTabStreaming(jobId),
        });

      } else {
        // ── New mode: create tab first, then stream into it ────────────
        const tempJobId = 'job_' + Date.now();
        const newItem = {
          jobId:                tempJobId,
          prompt_given_by_user: trimmedPrompt,
          slm_generated:        '',
          llm_generated:        null,
          isStreaming:          true,
          streamingText:        '',
          timestamp:            new Date().toISOString(),
          chatHistory:          [],
        };

        // Open tab immediately — user sees the streaming UI right away
        useAppStore.getState().addToHistory(newItem);
        useAppStore.getState().openHistoryItem(newItem);

        await api.streamJobDescription(trimmedPrompt, useLLM, {
          onChunk: (text) => useAppStore.getState().setTabStreaming(tempJobId, text),
          onDone:  (full, source) => {
            const isLLM = source === 'LLM';
            useAppStore.getState().updateTabContent(tempJobId, full, isLLM);
            useAppStore.getState().setTabDisplay(tempJobId, full, null);
            useAppStore.getState().clearTabStreaming(tempJobId);
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
    ? (activeTab.prompt_given_by_user?.slice(0, 44) ?? '') +
    (activeTab.prompt_given_by_user?.length > 44 ? '…' : '')
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

        {/* <div className="chat-options">
          <label className="chat-option-label" htmlFor="model-slm">
            <input id="model-slm" type="radio" name="model_choice"
              checked={!useLLM} onChange={() => setUseLLM(false)} />
            <Cpu size={12} />
            Local SLM
          </label>
          <label className="chat-option-label" htmlFor="model-llm">
            <input id="model-llm" type="radio" name="model_choice"
              checked={useLLM} onChange={() => setUseLLM(true)} />
            <Bot size={12} />
            External LLM
          </label>
        </div> */}
      </form>
    </div>
  );
}
