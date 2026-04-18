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
  const [saving,    setSaving]    = useState(false);
  const [saved,     setSaved]     = useState(false);
  const [enhancing, setEnhancing] = useState(false);
  const [exportOpen, setExportOpen] = useState(false);

  const { updateTabContent, addVersion, setTabDisplay } = useAppStore();

  // Show displayContent (version preview) if set, else latest generated
  const initialMarkdown = tab.displayContent
    ?? tab.llm_generated
    ?? tab.slm_generated
    ?? '';

  const sourceLabel = tab.llm_generated ? 'LLM' : 'SLM';
  const hasBoth     = !!(tab.slm_generated && tab.llm_generated);
  const versions    = tab.versions || [];

  // Version label in header
  const versionLabel = tab.activeVersionIdx != null
    ? `v${tab.activeVersionIdx + 1} · ${versions[tab.activeVersionIdx]?.label ?? ''}`
    : `Current · ${sourceLabel}`;

  // ── Smart save: only snapshot if content actually changed ──────────────────
  const handleSave = async () => {
    if (!ref.current) return;
    setSaving(true);
    const md = ref.current.getMarkdown().trim();

    // Detect what kind of content this is:
    // - If it matches llm_generated → LLM output (no edit)
    // - If it matches slm_generated → SLM output (no edit)
    // - Otherwise → user has edited the content
    const slmBase = (tab.slm_generated ?? '').trim();
    const llmBase = (tab.llm_generated ?? '').trim();
    const isUserEdit = md !== slmBase && md !== llmBase;
    const isLLM      = !!tab.llm_generated;

    const label  = isUserEdit ? 'User Edited'  : (isLLM ? 'LLM Output' : 'SLM Output');
    const source = isUserEdit ? 'USER'         : (isLLM ? 'LLM'        : 'SLM');

    // Only snapshot if content differs from the last saved version
    const lastVersion = versions[versions.length - 1];
    const isDirty = !lastVersion || lastVersion.content.trim() !== md;

    if (isDirty) {
      // User edits go into a separate field, don't overwrite slm/llm outputs
      if (isUserEdit) {
        useAppStore.getState().updateTabContent(tab.jobId, md, false); // store in slm slot as fallback
      }
      addVersion(tab.jobId, {
        label,
        source,
        content:   md,
        timestamp: new Date().toISOString(),
      });
    }

    try {
      await api.saveUserEdits({
        jobId:                tab.jobId,
        prompt_given_by_user: tab.prompt_given_by_user,
        slm_output:           tab.slm_generated || '',
        llm_output:           tab.llm_generated || null,
        user_edited:          isUserEdit ? md : null,  // only set when user actually changed text
      });
      setSaved(true);
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
    try {
      const current  = ref.current.getMarkdown();
      const response = await api.generateJobDescription(
        `Enhance this job description:\n\n${current}`, true
      );
      if (!tab.slm_generated && current) updateTabContent(tab.jobId, current, false);
      updateTabContent(tab.jobId, response.generatedMarkdown, true);
      const newVersionIdx = versions.length; // index BEFORE addVersion
      addVersion(tab.jobId, {
        label:     'LLM Enhanced',
        source:    'LLM',
        content:   response.generatedMarkdown,
        timestamp: new Date().toISOString(),
      });
      // Clear version preview so editor shows fresh LLM content
      setTabDisplay(tab.jobId, response.generatedMarkdown, newVersionIdx);
    } catch (err) {
      console.error(err);
      alert('Enhancement failed.');
    } finally {
      setEnhancing(false);
    }
  };

  // ── Export ────────────────────────────────────────────────────────────────
  const exportMD = useCallback(() => {
    const md   = ref.current?.getMarkdown() || '';
    const blob = new Blob([md], { type: 'text/markdown' });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href     = url;
    a.download = `job-description-${tab.jobId}.md`;
    a.click();
    URL.revokeObjectURL(url);
    setExportOpen(false);
  }, [tab.jobId]);

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
          {/* Enhance */}
          <button
            id="enhance-llm-btn"
            className="btn-enhance"
            onClick={handleEnhance}
            disabled={enhancing}
          >
            <Sparkles size={13} className={enhancing ? 'spin-icon' : ''} />
            {enhancing ? 'Enhancing…' : 'Enhance with LLM'}
          </button>

          {/* Save */}
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

          {/* Export dropdown */}
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
          key={tab.displayKey ?? (tab.llm_generated ? 'llm' : 'slm')}
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
