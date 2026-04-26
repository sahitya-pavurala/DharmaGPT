# DharmaGPT Mobile — Setup & Publishing Guide

## Prerequisites

- Node.js 18+ installed
- Expo CLI: `npm install -g expo-cli eas-cli`
- Apple Developer account ($99/yr) for iOS
- Google Play Developer account ($25 one-time) for Android
- Expo account (free): https://expo.dev

---

## 1. Install dependencies

```bash
cd mobile
npm install
```

---

## 2. Set your API URL

```bash
cp .env.example .env.local
# Edit .env.local and set EXPO_PUBLIC_API_URL to your beta server
```

---

## 3. Run locally

```bash
npx expo start
```

- Press `i` for iOS Simulator (requires Xcode on Mac)
- Press `a` for Android Emulator (requires Android Studio)
- Scan the QR code with the **Expo Go** app on your phone for instant preview

---

## 4. Add app assets (required before store submission)

Place these files in `mobile/assets/`:

| File | Size | Notes |
|------|------|-------|
| `icon.png` | 1024×1024 | No alpha channel (App Store requirement) |
| `splash.png` | 1284×2778 | Splash screen background |
| `adaptive-icon.png` | 1024×1024 | Android adaptive icon foreground |
| `favicon.png` | 48×48 | Web favicon |

Design tip: Use a saffron (#FF6B00) background with the DharmaGPT logo centered.

---

## 5. Create your Expo project

```bash
npx eas build:configure
# Follow prompts — log in with your Expo account
```

---

## 6. Build for stores

### iOS (TestFlight first, then App Store)
```bash
# Internal preview build
npx eas build --platform ios --profile preview

# Production build
npx eas build --platform ios --profile production
```

### Android
```bash
npx eas build --platform android --profile production
```

---

## 7. Submit to stores

### App Store (iOS)
```bash
npx eas submit --platform ios
# EAS will ask for your Apple ID and upload automatically
```

You'll need to complete the App Store Connect listing manually:
- App description (see below)
- Screenshots (at least 3 for iPhone 6.7")
- Privacy policy URL: https://dharmagpt.com/privacy
- Category: Education → Reference
- Age rating: 4+

### Google Play (Android)
```bash
npx eas submit --platform android
```

---

## App Store Listing Copy

**Name:** DharmaGPT

**Subtitle:** Wisdom from Sacred Scriptures

**Description:**
DharmaGPT brings the timeless wisdom of Hindu sacred texts to your fingertips. Ask questions about dharma, karma, and righteous living — and receive answers grounded in the Ramayana, Mahabharata, and Puranas, with full citations.

**Features:**
- 4 response modes: Guidance, Story, Scholar, and Children
- Answers in English, Telugu, Hindi, Tamil, Kannada, Malayalam, and Sanskrit
- Every answer cites the exact scripture, kanda, chapter, and verse
- Conversation history across sessions
- Thumbs up/down feedback to improve accuracy

**Keywords:** dharma, ramayana, mahabharata, hindu, scripture, vedic, spirituality, gita, karma, bhagavad

---

## Required before App Store submission

- [ ] App icon (1024×1024, no alpha)
- [ ] 3+ iPhone screenshots
- [ ] Privacy Policy URL live at your domain
- [ ] EXPO_PUBLIC_API_URL pointing to production server
- [ ] Apple Developer account enrolled
- [ ] Bundle ID registered: `com.dharmagpt.app`

---

## Project structure

```
mobile/
├── app/
│   ├── _layout.tsx          # Root layout + splash
│   └── (tabs)/
│       ├── _layout.tsx      # Tab bar
│       ├── index.tsx        # Chat screen
│       └── settings.tsx     # Settings screen
├── components/
│   ├── MessageBubble.tsx    # Chat message with sources + feedback
│   ├── ModeSelector.tsx     # Horizontal mode chips
│   └── SourcesPanel.tsx     # Expandable citation cards
├── services/
│   └── api.ts               # Typed API client
├── store/
│   ├── chatStore.ts         # Chat state (Zustand)
│   └── settingsStore.ts     # Persisted settings
├── constants/
│   └── theme.ts             # Colors, fonts, spacing
├── assets/                  # ← Add your icon/splash here
├── app.json                 # Expo config
├── eas.json                 # EAS Build config
└── .env.example             # Copy to .env.local
```
