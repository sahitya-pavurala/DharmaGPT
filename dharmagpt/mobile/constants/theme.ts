// DharmaGPT Design Tokens — spiritual saffron palette

export const Colors = {
  // Brand
  saffron: '#FF6B00',
  saffronLight: '#FF8C3A',
  saffronDark: '#CC5500',
  gold: '#FFB830',
  lotus: '#C2185B',

  // Neutrals
  cream: '#FFF8F0',
  parchment: '#FDF3E3',
  sandal: '#E8D5B7',
  ash: '#9E9E9E',

  // Dark mode
  darkBg: '#1A1208',
  darkCard: '#2C1F0E',
  darkBorder: '#3D2B12',

  // Semantic
  white: '#FFFFFF',
  black: '#000000',
  error: '#D32F2F',
  success: '#388E3C',

  // Mode badge colors
  modes: {
    guidance: '#FF6B00',
    story: '#7B1FA2',
    children: '#0288D1',
    scholar: '#388E3C',
  },
} as const;

export const Fonts = {
  regular: 'System',
  medium: 'System',
  bold: 'System',
  sizes: {
    xs: 11,
    sm: 13,
    md: 15,
    lg: 17,
    xl: 20,
    xxl: 24,
    display: 30,
  },
} as const;

export const Spacing = {
  xs: 4,
  sm: 8,
  md: 16,
  lg: 24,
  xl: 32,
  xxl: 48,
} as const;

export const Radius = {
  sm: 8,
  md: 12,
  lg: 18,
  xl: 24,
  pill: 999,
} as const;

export const MODES = [
  {
    key: 'guidance' as const,
    label: 'Guidance',
    icon: 'compass',
    description: 'Practical wisdom for daily life',
  },
  {
    key: 'story' as const,
    label: 'Story',
    icon: 'book-open',
    description: 'Narrative retelling of scripture',
  },
  {
    key: 'children' as const,
    label: 'Children',
    icon: 'star',
    description: 'Simple explanations for young minds',
  },
  {
    key: 'scholar' as const,
    label: 'Scholar',
    icon: 'graduation-cap',
    description: 'In-depth analysis with citations',
  },
] as const;

export const LANGUAGES = [
  { code: 'en', label: 'English', native: 'English' },
  { code: 'te', label: 'Telugu', native: 'తెలుగు' },
  { code: 'hi', label: 'Hindi', native: 'हिन्दी' },
  { code: 'ta', label: 'Tamil', native: 'தமிழ்' },
  { code: 'kn', label: 'Kannada', native: 'ಕನ್ನಡ' },
  { code: 'ml', label: 'Malayalam', native: 'മലയാളം' },
  { code: 'sa', label: 'Sanskrit', native: 'संस्कृतम्' },
] as const;
