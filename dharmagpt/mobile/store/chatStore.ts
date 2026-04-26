// Global chat state using Zustand
import { create } from 'zustand';
import { ChatMessage, QueryMode, SourceChunk, QueryResponse } from '@/services/api';

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  sources?: SourceChunk[];
  query_id?: string;
  mode?: QueryMode;
  feedback?: 'up' | 'down';
  timestamp: number;
}

interface ChatState {
  messages: Message[];
  isLoading: boolean;
  mode: QueryMode;
  language: string;
  filterSection: string | undefined;

  addMessage: (msg: Omit<Message, 'id' | 'timestamp'>) => string;
  updateMessage: (id: string, updates: Partial<Message>) => void;
  setLoading: (v: boolean) => void;
  setMode: (mode: QueryMode) => void;
  setLanguage: (lang: string) => void;
  setFilterSection: (section: string | undefined) => void;
  clearHistory: () => void;

  /** Returns last N messages as ChatMessage[] for the API history param */
  getApiHistory: (n?: number) => ChatMessage[];
}

export const useChatStore = create<ChatState>((set, get) => ({
  messages: [],
  isLoading: false,
  mode: 'guidance',
  language: 'en',
  filterSection: undefined,

  addMessage: (msg) => {
    const id = Math.random().toString(36).slice(2);
    set((s) => ({
      messages: [...s.messages, { ...msg, id, timestamp: Date.now() }],
    }));
    return id;
  },

  updateMessage: (id, updates) =>
    set((s) => ({
      messages: s.messages.map((m) => (m.id === id ? { ...m, ...updates } : m)),
    })),

  setLoading: (v) => set({ isLoading: v }),
  setMode: (mode) => set({ mode }),
  setLanguage: (language) => set({ language }),
  setFilterSection: (filterSection) => set({ filterSection }),
  clearHistory: () => set({ messages: [] }),

  getApiHistory: (n = 10) => {
    const msgs = get().messages.slice(-n * 2);
    return msgs.map((m) => ({ role: m.role, content: m.content }));
  },
}));
