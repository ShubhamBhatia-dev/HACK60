import { create } from 'zustand';
import { api } from '../api/client';

// Re-hydrate user from localStorage so refresh keeps you logged in.
// The JWT itself is stored separately via api helpers.
function loadUser() {
  try { return JSON.parse(localStorage.getItem('slm_user')); } catch { return null; }
}

export const useAppStore = create((set) => ({
  user: loadUser(),
  history: [],
  openTabs: [],     // { job_id, prompt, slm_output, llm_output, chatHistory[], versions[] }
  activeTabId: null,
  selectedModel: 'qwen',  // active SLM model key

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

  // Sync a history entry and move to top
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
      activeTabId: item.job_id
    };
  }),

  closeTab: (job_id) => set((state) => {
    const newTabs = state.openTabs.filter(t => t.job_id !== job_id);
    let newActive = state.activeTabId;
    if (state.activeTabId === job_id) {
      newActive = newTabs.length > 0 ? newTabs[newTabs.length - 1].job_id : null;
    }
    return { openTabs: newTabs, activeTabId: newActive };
  }),

  // Simple: set slm_output or llm_output — that's it
  updateTabContent: (job_id, content, isLLM = false) => set((state) => ({
    openTabs: state.openTabs.map(tab => {
      if (tab.job_id !== job_id) return tab;
      return isLLM
        ? { ...tab, llm_output: content }
        : { ...tab, slm_output: content };
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

  // View a version (idx = number) or go back to current (idx = null)
  setTabDisplay: (job_id, versionIdx) => set((state) => ({
    openTabs: state.openTabs.map(tab => {
      if (tab.job_id !== job_id) return tab;
      return {
        ...tab,
        activeVersionIdx: versionIdx,
        displayKey:       ((tab.displayKey) || 0) + 1,
      };
    })
  })),

  // Streaming
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
