import React, { useRef, useState, useCallback } from 'react';
import {
  MDXEditor,
  headingsPlugin,
  listsPlugin,
  quotePlugin,
  thematicBreakPlugin,
  markdownShortcutPlugin,
  toolbarPlugin,
  UndoRedo,
  BoldItalicUnderlineToggles,
  BlockTypeSelect,
} from '@mdxeditor/editor';
import '@mdxeditor/editor/style.css';
import { Save, CheckCircle, Sparkles, Download, ChevronDown } from 'lucide-react';
import { api } from '../api/client';
import { useAppStore } from '../store/useAppStore';

export default function MarkdownEditor({ tab }) {
  const ref = useRef(null);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [enhancing, setEnhancing] = useState(false);
  const [exportOpen, setExportOpen] = useState(false);

  const { updateTabContent, addVersion, setTabDisplay, setTabStreaming, clearTabStreaming, updateHistoryItem, selectedModel } = useAppStore();

  const versions = tab.versions || [];

  // ── What to show ──────────────────────────────────────────────────────────
  // Version preview? → show that version's content
  // Otherwise       → show latest output (slm or llm, whichever exists)
  let initialMarkdown = '';
  if (tab.activeVersionIdx != null) {
    initialMarkdown = versions[tab.activeVersionIdx]?.content || '';
  } else {
    initialMarkdown = tab.slm_output || tab.llm_output || '';
  }

  const sourceLabel = tab.llm_output ? 'LLM' : 'SLM';
  const hasBoth = !!(tab.slm_output && tab.llm_output);

  const versionLabel = tab.activeVersionIdx != null
    ? `v${tab.activeVersionIdx + 1} · ${versions[tab.activeVersionIdx]?.label ?? ''}`
    : `Current · ${sourceLabel}`;

  // ── Smart save ────────────────────────────────────────────────────────────
  const normalize = (s) => (s ?? '').replace(/\s+/g, ' ').trim();

  const handleSave = async () => {
    if (!ref.current) return;
    setSaving(true);
    const md = ref.current.getMarkdown().trim();

    const slmBase = normalize(tab.slm_output);
    const llmBase = normalize(tab.llm_output);
    const mdNorm = normalize(md);

    const isUserEdit = mdNorm !== slmBase && mdNorm !== llmBase;

    const lastVersion = versions[versions.length - 1];
    const isDirty = !lastVersion || normalize(lastVersion.content) !== mdNorm;

    if (isDirty && isUserEdit) {
      addVersion(tab.job_id, {
        label: 'User Edited',
        source: 'USER',
        content: md,
        timestamp: new Date().toISOString(),
      });
    }

    try {
      await api.saveUserEdits({
        job_id: tab.job_id,
        prompt: tab.prompt,
        slm_output: tab.slm_output || '',
        llm_output: tab.llm_output || null,
        user_edited: isUserEdit ? md : null,
      });
      setSaved(true);
      updateHistoryItem(tab.job_id, {
        user_edited: isUserEdit ? md : null,
        updated_at: new Date().toISOString()
      });
      setTimeout(() => setSaved(false), 2500);
    } catch (err) {
      console.error(err);
      alert('Failed to save.');
    } finally {
      setSaving(false);
    }
  };

  // ── Enhance with LLM ──────────────────────────────────────────────────────
  const handleEnhance = async () => {
    if (!ref.current) return;
    setEnhancing(true);
    const job_id = tab.job_id;
    const currentMd = ref.current.getMarkdown();
    if (!tab.slm_output && currentMd) updateTabContent(job_id, currentMd, false);

    setTabStreaming(job_id, '');

    try {
      await api.streamJobDescription(
        tab.prompt,
        true,
        currentMd,
        selectedModel,
        {
          onChunk: (text) => setTabStreaming(job_id, text),
          onDone: (full) => {
            updateTabContent(job_id, full, true);
            addVersion(job_id, {
              label: 'LLM Enhanced',
              source: 'LLM',
              content: full,
              timestamp: new Date().toISOString(),
            });
            clearTabStreaming(job_id);
            // Bump displayKey so editor remounts with new content
            setTabDisplay(job_id, null);
            updateHistoryItem(job_id, {
              llm_output: full,
              updated_at: new Date().toISOString()
            });
            api.saveUserEdits({
              job_id, prompt: tab.prompt, slm_output: tab.slm_output, llm_output: full,
              user_edited: tab.user_edited || null
            }).catch(err => console.error('Auto-save failed:', err));
          },
          onError: (err) => {
            clearTabStreaming(job_id);
            throw err;
          },
        }
      );
    } catch (err) {
      console.error(err);
      alert('Enhancement failed.');
    } finally {
      setEnhancing(false);
    }
  };

  // ── Export ────────────────────────────────────────────────────────────────
  const exportMD = useCallback(() => {
    const md = ref.current?.getMarkdown() || '';
    const blob = new Blob([md], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `job-description-${tab.job_id}.md`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    setExportOpen(false);
  }, [tab.job_id]);

  const exportPDF = useCallback(() => {
    window.print();
    setExportOpen(false);
  }, []);

  return (
    <div className="editor-content">
      {/* ── Header ── */}
      <div className="editor-header">
        <div className="editor-title-block">
          <h3>Job Description</h3>
          <div className="editor-meta">
            <span className={`meta-badge ${sourceLabel === 'LLM' ? 'meta-badge--llm' : ''}`}>
              {versionLabel}
            </span>
            {hasBoth && <span className="meta-both">· Both versions saved</span>}
          </div>
        </div>

        <div className="editor-actions">
          <button
            id="enhance-llm-btn"
            className="btn-enhance"
            onClick={handleEnhance}
            disabled={enhancing}
          >
            <Sparkles size={13} className={enhancing ? 'spin-icon' : ''} />
            {enhancing ? 'Enhancing…' : 'Enhance with LLM'}
          </button>

          <button
            id="save-edits-btn"
            className="btn-secondary"
            onClick={handleSave}
            disabled={saving}
          >
            {saved
              ? <><CheckCircle size={13} style={{ color: 'var(--success)' }} />Saved</>
              : <><Save size={13} />{saving ? 'Saving…' : 'Save'}</>}
          </button>

          <div className="export-wrap" onBlur={() => setExportOpen(false)} tabIndex={-1}>
            <button
              id="export-btn"
              className="btn-secondary export-trigger"
              onClick={() => setExportOpen(v => !v)}
            >
              <Download size={13} />
              Export
              <ChevronDown size={11} />
            </button>
            {exportOpen && (
              <div className="export-dropdown">
                <button onClick={exportMD}>Raw Markdown (.md)</button>
                <button onClick={exportPDF}>PDF (Print dialog)</button>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* ── MDX Editor ── */}
      <div className="mdxeditor-wrapper">
        <MDXEditor
          ref={ref}
          markdown={initialMarkdown}
          plugins={[
            headingsPlugin(),
            listsPlugin(),
            quotePlugin(),
            thematicBreakPlugin(),
            markdownShortcutPlugin(),
            toolbarPlugin({
              toolbarContents: () => (
                <>
                  <UndoRedo />
                  <BoldItalicUnderlineToggles />
                  <BlockTypeSelect />
                </>
              ),
            }),
          ]}
          contentEditableClassName="editor-body"
        />
      </div>
    </div>
  );
}
