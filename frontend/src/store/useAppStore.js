import { create } from 'zustand';
import { api } from '../api/client';

function loadUser() {
  try { return JSON.parse(localStorage.getItem('slm_user')); } catch { return null; }
}

export const useAppStore = create((set) => ({
  user: loadUser(),
  history: [],
  openTabs: [],
  activeTabId: null,
  selectedModel: 'qwen',

  setSelectedModel: (model) => set({ selectedModel: model }),

  setUser: (user) => {
    localStorage.setItem('slm_user', JSON.stringify(user));
    set({ user });
  },

  logout: () => {
    api.logout();
    localStorage.removeItem('slm_user');
    set({ user: null, openTabs: [], activeTabId: null, history: [] });
  },

  setHistory: (history) => set({ history }),

  addToHistory: (item) => set((state) => ({
    history: [item, ...state.history]
  })),

  updateHistoryItem: (job_id, updates) => set((state) => {
    const item = state.history.find(i => i.job_id === job_id);
    if (!item) return state;
    return { history: [{ ...item, ...updates }, ...state.history.filter(i => i.job_id !== job_id)] };
  }),

  openHistoryItem: (item) => set((state) => {
    const existing = state.openTabs.find(t => t.job_id === item.job_id);
    if (existing) return { activeTabId: item.job_id };
    return {
      openTabs: [...state.openTabs, { ...item, chatHistory: item.chatHistory || [] }],
      activeTabId: item.job_id,
    };
  }),

  closeTab: (job_id) => set((state) => {
    const newTabs = state.openTabs.filter(t => t.job_id !== job_id);
    const newActive = state.activeTabId === job_id
      ? (newTabs.length > 0 ? newTabs[newTabs.length - 1].job_id : null)
      : state.activeTabId;
    return { openTabs: newTabs, activeTabId: newActive };
  }),

  // Called when a model finishes generating.
  // Sets slm_output / llm_output, sets displayOutput (what editor shows),
  // and bumps displayKey to force MDXEditor remount.
  updateTabContent: (job_id, content, isLLM = false) => set((state) => ({
    openTabs: state.openTabs.map(tab => {
      if (tab.job_id !== job_id) return tab;
      return {
        ...tab,
        ...(isLLM ? { llm_output: content } : { slm_output: content }),
        displayOutput: content,                   // ← editor reads this
        previewContent: null,                     // ← clear any version preview
        activeVersionIdx: null,
        displayKey: ((tab.displayKey) || 0) + 1, // ← forces MDXEditor remount
      };
    })
  })),

  appendTabChatMessage: (job_id, message) => set((state) => ({
    openTabs: state.openTabs.map(tab => {
      if (tab.job_id !== job_id) return tab;
      return { ...tab, chatHistory: [...(tab.chatHistory || []), message] };
    })
  })),

  setActiveTab: (job_id) => set({ activeTabId: job_id }),

  addVersion: (job_id, version) => set((state) => ({
    openTabs: state.openTabs.map(tab => {
      if (tab.job_id !== job_id) return tab;
      return { ...tab, versions: [...(tab.versions || []), version] };
    })
  })),

  // View a specific version (idx = number) — sets previewContent, not displayOutput.
  // Restore current (idx = null)         — clears previewContent, shows displayOutput again.
  // Always bumps displayKey to remount editor.
  setTabDisplay: (job_id, versionIdx) => set((state) => ({
    openTabs: state.openTabs.map(tab => {
      if (tab.job_id !== job_id) return tab;
      return {
        ...tab,
        activeVersionIdx: versionIdx,
        previewContent: versionIdx != null
          ? ((tab.versions ?? [])[versionIdx]?.content ?? '')
          : null,                                // null = show displayOutput (the live output)
        displayKey: ((tab.displayKey) || 0) + 1,
      };
    })
  })),

  // Streaming — just update the live preview text
  setTabStreaming: (job_id, text) => set((state) => ({
    openTabs: state.openTabs.map(tab => {
      if (tab.job_id !== job_id) return tab;
      return { ...tab, isStreaming: true, streamingText: text, activeVersionIdx: null };
    })
  })),

  clearTabStreaming: (job_id) => set((state) => ({
    openTabs: state.openTabs.map(tab => {
      if (tab.job_id !== job_id) return tab;
      return { ...tab, isStreaming: false, streamingText: '' };
    })
  })),
}));
