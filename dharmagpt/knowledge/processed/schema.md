# DharmaGPT Corpus — JSONL Schema

Each line in `knowledge/processed/*.jsonl` is a self-contained JSON object representing one **chunk** of scripture or commentary.

---

## Field Reference

| Field | Type | Required | Description |
|---|---|---|---|
| `id` | string | ✅ | Unique chunk ID. Format: `{source_key}_{kanda_slug}_{sarga:04d}_{chunk:03d}` |
| `text` | string | ✅ | The chunk text. 80–300 words. In the `language` below. |
| `text_te` | string | ❌ | Telugu translation or original (if `language != "te"`) |
| `text_en` | string | ❌ | English translation (if `language != "en"`) |
| `source` | string | ✅ | Canonical source key. See Sources table below. |
| `kanda` | string | ❌ | Kanda/Parva/Chapter name. e.g. `"Sundara Kanda"`, `"Bhishma Parva"` |
| `sarga` | int | ❌ | Sarga / chapter number within kanda |
| `verse_start` | int | ❌ | First verse number in chunk |
| `verse_end` | int | ❌ | Last verse number in chunk |
| `citation` | string | ✅ | Human-readable cite. e.g. `"Valmiki Ramayana, Sundara Kanda, Sarga 35, Verses 1–8"` |
| `language` | string | ✅ | Primary language of `text`. ISO 639-1: `"sa"` (Sanskrit), `"te"`, `"en"`, `"hi"` |
| `source_type` | string | ✅ | `"text"` \| `"commentary"` \| `"audio_transcript"` |
| `tags` | list[str] | ✅ | Semantic tags for filtering. See Tags table. |
| `topics` | list[str] | ❌ | Free-form concepts: `["dharma", "devotion", "bhakti"]` |
| `characters` | list[str] | ❌ | Named entities: `["Rama", "Hanuman", "Sita"]` |
| `is_shloka` | bool | ✅ | `true` if chunk is metered Sanskrit verse |
| `url` | string | ❌ | Source URL if online. e.g. valmikiramayan.net |
| `notes` | string | ❌ | Curator notes: textual variants, translation choices |

---

## Canonical Source Keys

| `source` value | Full name |
|---|---|
| `valmiki_ramayana` | Valmiki Ramayana (7 Kandas) |
| `mahabharata` | Mahabharata (18 Parvas) |
| `bhagavad_gita` | Bhagavad Gita (within Bhishma Parva) |
| `bhagavata_purana` | Bhagavata Purana (12 Skandhas) |
| `vishnu_purana` | Vishnu Purana |
| `brihadaranyaka_upanishad` | Brihadaranyaka Upanishad |
| `chandogya_upanishad` | Chandogya Upanishad |
| `katha_upanishad` | Katha Upanishad |
| `isha_upanishad` | Isha Upanishad |
| `mandukya_upanishad` | Mandukya Upanishad |
| `mundaka_upanishad` | Mundaka Upanishad |

---

## Allowed Tags

`devotion` · `dharma` · `karma` · `moksha` · `bhakti` · `jnana` · `vairagya` · `ethics` · `courage` · `surrender` · `grief` · `duty` · `war` · `family` · `guru_disciple` · `creation` · `cosmology` · `ritual` · `pilgrimage` · `shloka` · `story` · `upadesha` (teaching) · `stotram`

---

## Example Record

```json
{
  "id": "valmiki_ramayana_sundara_0035_001",
  "text": "tataH suvarNapuSpANi vRkSA rukmavibhUSitAH | ...",
  "text_en": "Then the trees adorned with golden flowers and decorated with gold ...",
  "source": "valmiki_ramayana",
  "kanda": "Sundara Kanda",
  "sarga": 35,
  "verse_start": 1,
  "verse_end": 8,
  "citation": "Valmiki Ramayana, Sundara Kanda, Sarga 35, Verses 1–8",
  "language": "sa",
  "source_type": "text",
  "tags": ["devotion", "story"],
  "topics": ["Lanka", "Hanuman searching for Sita"],
  "characters": ["Hanuman", "Sita"],
  "is_shloka": true,
  "url": "https://www.valmikiramayan.net/utf8/sundar/sarga35/sundarasans35.htm"
}
```
