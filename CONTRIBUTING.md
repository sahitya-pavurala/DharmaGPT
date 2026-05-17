# Contributing to DharmaGPT

First — thank you. This project exists to make the wisdom of Hindu sacred texts accessible to everyone through technology. Every contribution, large or small, matters.

---

## Ways to Contribute

### 1. Sanskrit & Textual Scholars
- Verify that AI-generated citations are accurate
- Flag hallucinated or incorrect verse references
- Add missing texts to the corpus (Puranas, Smritis, Agamas)
- Improve transliterations and translations

### 2. Developers
- Backend (FastAPI / Python): RAG pipeline, audio chunking, API routes
- Mobile (React Native / Expo): UI components, audio recording, offline support
- Scripts: scraper improvements, embedding pipeline, chunking strategies
- DevOps: deployment, CI/CD, monitoring

### 3. Audio Contributors
- Record or donate audio files: pravachanams, chantings, discourses
- Verify Sarvam AI transcriptions for accuracy
- Add metadata JSON for audio files (kanda, sarga, speaker, language)

### 4. Language Contributors
- Help expand support for regional language responses
- Translate the UI into Indian languages
- Test Sarvam AI output quality for specific languages

### 5. Community
- Write documentation and usage guides
- Report bugs or factual errors
- Share the project with scholars and spiritual communities

---

## Getting Started

```bash
git clone https://github.com/ShambaviLabs/DharmaGPT.git
cd DharmaGPT/dharmagpt
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

---

## Pull Request Guidelines

1. **One PR per concern** — don't mix a bug fix with a new feature
2. **Reference the issue** — every PR should link to an open issue
3. **Test your changes** — run `make test-unit` before opening a PR
4. **Citation accuracy** — if you add text corpus content, cite the source clearly
5. **No fabricated content** — never add invented shlokas or incorrect attributions

---

## Reporting Factual Errors

This is the most important category of issue. If DharmaGPT gives an incorrect citation, invents a verse, or misattributes content:

1. Open an issue with label `factual-error`
2. Include: the query, the wrong answer, the correct text/citation
3. Link to the source (valmikiramayan.net, sacred-texts.com, etc.)

We treat factual errors as critical bugs.

---

## Code of Conduct

- Treat all traditions and interpretations with respect
- No sectarian debates — this is a knowledge tool, not a theology forum
- Be constructive in code reviews
- Credit your sources

---

## Contact

Open an issue or discussion on GitHub. We're especially looking to connect with:
- Sanskrit scholars and Vedic institutions
- Contributors from the TTD, Ramakrishna Mission, and similar organizations
- Indian language AI researchers

---

*Om Tat Sat.*
