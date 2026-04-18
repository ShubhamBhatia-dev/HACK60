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
  openTabs: [],     // { jobId, prompt_given_by_user, slm_generated, llm_generated, chatHistory[] }
  activeTabId: null,

  setUser: (user) => {
    localStorage.setItem('slm_user', JSON.stringify(user));
    set({ user });
  },
  logout: () => {
    api.logout();                                    // clears JWT from localStorage
    localStorage.removeItem('slm_user');
    set({ user: null, openTabs: [], activeTabId: null, history: [] });
  },

  setHistory: (history) => set({ history }),

  // Add a new item to history
  addToHistory: (item) => set((state) => ({
    history: [item, ...state.history]
  })),

  // Open an item from history into a new tab (or switch to it if already open)
  openHistoryItem: (item) => set((state) => {
    const existingTab = state.openTabs.find(tab => tab.jobId === item.jobId);
    if (existingTab) {
      return { activeTabId: item.jobId };
    }
    return {
      openTabs: [...state.openTabs, { ...item, chatHistory: item.chatHistory || [] }],
      activeTabId: item.jobId
    };
  }),

  // Close a specific tab
  closeTab: (jobId) => set((state) => {
    const newTabs = state.openTabs.filter(tab => tab.jobId !== jobId);
    let newActiveId = state.activeTabId;
    if (state.activeTabId === jobId) {
      newActiveId = newTabs.length > 0 ? newTabs[newTabs.length - 1].jobId : null;
    }
    return { openTabs: newTabs, activeTabId: newActiveId };
  }),

  // Update editor content of a tab
  updateTabContent: (jobId, content, isLLM = false) => set((state) => ({
    openTabs: state.openTabs.map(tab => {
      if (tab.jobId !== jobId) return tab;
      return isLLM
        ? { ...tab, llm_generated: content }
        : { ...tab, slm_generated: content };
    })
  })),

  // Append a chat message to a specific tab's chat history
  appendTabChatMessage: (jobId, message) => set((state) => ({
    openTabs: state.openTabs.map(tab => {
      if (tab.jobId !== jobId) return tab;
      return { ...tab, chatHistory: [...(tab.chatHistory || []), message] };
    })
  })),

  // Set active tab
  setActiveTab: (jobId) => set({ activeTabId: jobId }),

  // Add a version snapshot to a tab's versions array
  // version: { label, source, content, timestamp }
  addVersion: (jobId, version) => set((state) => ({
    openTabs: state.openTabs.map(tab => {
      if (tab.jobId !== jobId) return tab;
      return { ...tab, versions: [...(tab.versions || []), version] };
    })
  })),

  // Load a specific version into the editor (by index, or null for "current")
  setTabDisplay: (jobId, content, versionIdx) => set((state) => ({
    openTabs: state.openTabs.map(tab => {
      if (tab.jobId !== jobId) return tab;
      return {
        ...tab,
        displayContent:   content,
        activeVersionIdx: versionIdx,
        displayKey:       ((tab.displayKey) || 0) + 1,
      };
    })
  })),

  // Update live streaming text (called per chunk during SSE)
  setTabStreaming: (jobId, text) => set((state) => ({
    openTabs: state.openTabs.map(tab => {
      if (tab.jobId !== jobId) return tab;
      return { ...tab, isStreaming: true, streamingText: text };
    })
  })),

  // Clear streaming state when SSE is complete
  clearTabStreaming: (jobId) => set((state) => ({
    openTabs: state.openTabs.map(tab => {
      if (tab.jobId !== jobId) return tab;
      return { ...tab, isStreaming: false, streamingText: '' };
    })
  })),
}));
