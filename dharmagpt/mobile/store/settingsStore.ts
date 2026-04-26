import { create } from 'zustand';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { QueryMode } from '@/services/api';

interface SettingsState {
  language: string;
  defaultMode: QueryMode;
  showSources: boolean;
  hapticFeedback: boolean;
  loaded: boolean;

  setLanguage: (lang: string) => void;
  setDefaultMode: (mode: QueryMode) => void;
  setShowSources: (v: boolean) => void;
  setHapticFeedback: (v: boolean) => void;
  loadSettings: () => Promise<void>;
}

const STORAGE_KEY = 'dharmagpt_settings';

export const useSettingsStore = create<SettingsState>((set, get) => ({
  language: 'en',
  defaultMode: 'guidance',
  showSources: true,
  hapticFeedback: true,
  loaded: false,

  setLanguage: (language) => {
    set({ language });
    AsyncStorage.setItem(STORAGE_KEY, JSON.stringify({ ...get(), language }));
  },
  setDefaultMode: (defaultMode) => {
    set({ defaultMode });
    AsyncStorage.setItem(STORAGE_KEY, JSON.stringify({ ...get(), defaultMode }));
  },
  setShowSources: (showSources) => {
    set({ showSources });
    AsyncStorage.setItem(STORAGE_KEY, JSON.stringify({ ...get(), showSources }));
  },
  setHapticFeedback: (hapticFeedback) => {
    set({ hapticFeedback });
    AsyncStorage.setItem(STORAGE_KEY, JSON.stringify({ ...get(), hapticFeedback }));
  },

  loadSettings: async () => {
    try {
      const raw = await AsyncStorage.getItem(STORAGE_KEY);
      if (raw) {
        const saved = JSON.parse(raw);
        set({ ...saved, loaded: true });
      } else {
        set({ loaded: true });
      }
    } catch {
      set({ loaded: true });
    }
  },
}));
