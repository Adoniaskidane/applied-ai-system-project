# 🎧 Model Card: TuneJudge

## 1. Model Overview

**TuneJudge** recommends songs from a plain-English request ("calm music for studying", "high energy for a party") and explains each pick.

It combines three parts:

| Part | What it does | Type |
|---|---|---|
| **LLM parser** | Turns the request into a taste profile: genre, mood, energy (0–1), likes acoustic | AI: Claude, Gemini or OpenAI, with structured output |
| **Rule-based scorer** | Ranks every candidate song against that profile | Deterministic, about 30 lines of Python |
| **LLM explainer** | Writes a one-line "why" for each pick | AI: the same provider, structured output |

Retrieval (TF-IDF over a knowledge base of notes and over the catalog) feeds both AI steps, and three guardrails check the input, the parsed profile and the final explanation. Full architecture: [README](README.md#architecture-overview).

---

## 2. Intended Use

**Intended for:**
- A portfolio and course project showing how to combine an LLM with retrieval, deterministic ranking and guardrails.
- Exploring "what if" questions about recommenders: how weights, data gaps and wording change results.

**Not intended for:**
- A production music service. The catalog is 945 songs, English-only, with approximate mood labels.
- Judging whether a song, artist or genre is *good*. A low score only means "not a match for this profile."
- Anything with real stakes, like paying artists or curating music for an event without human review.

---

## 3. Data

- **Catalog:** 945 real, English-language songs in `data/songs.csv`, from the public [Spotify Tracks Dataset](https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset). Each song has a Spotify ID, genre, mood, energy, tempo, valence, danceability and acousticness.
- **16 genres:** from 73 songs (lofi) down to 22 (reggae). Most genres have about 70, but r&b has 32, hip hop 38 and classical 48.
- **16 moods.** Songs 1–19 have moods I assigned by hand. Every other song gets the mood of its nearest hand-labeled song, compared on energy, valence, danceability and acousticness.
- **Cleaning** (`scripts/build_catalog.py`): Latin removed; vocal tracks kept only if they read as English (title and artist heuristics plus a hand-reviewed exclusion list), instrumentals kept; duplicates removed by Spotify ID and by title + lead artist; the most popular tracks kept per genre.
- **Knowledge base:** `data/knowledge/music_notes.md`, short notes on every genre, mood and common activity (studying, workout, night drive…), built from catalog averages.

---

## 4. How It Works

1. **Input guardrail** blocks empty, very long (over 500 characters), wordless and prompt-injection requests before any API call.
2. **Retrieve notes** that match the request.
3. **The LLM parses** the request into a profile, or flags it as off-topic.
4. **Profile guardrail** snaps genre and mood to catalog values, clamps energy to 0–1 and fixes types.
5. **Retrieve 120 candidate songs and score them:**

   | Rule | Points (current defaults) |
   |---|---|
   | Genre matches (related genre earns half) | 1.0 |
   | Mood matches | 1.5 |
   | Energy closeness: `3.0 × (1 − abs(song − target))` | up to 3.0 |
   | Acoustic fit | up to 0.5 |
   | **Maximum** | **6.0** |

6. **The LLM explains** the top picks, using only the retrieved songs and notes.
7. **Grounding check** rejects explanations that mention songs that weren't recommended, use wrong IDs or quote numbers not in the data, and falls back to the scorer's reasons.
8. **Log** every request to `logs/tunejudge.jsonl`.

Each result shows a fit level: Strong (≥ 75 % of max), Good (≥ 55 %) or Weak. The app warns when the requested genre or mood isn't in the catalog, or when no song reaches 55 %.

---

## 5. Evaluation Summary

Full results and reproducible commands are in the [README's Testing Summary](README.md#testing-summary) and [`evidence/`](evidence/).

| Check | Result |
|---|---|
| Automated tests (`pytest`, fake LLM, offline) | 77 / 77 pass |
| Knowledge-note retrieval hit@4 (12 golden queries) | 92 % (11 / 12) |
| Candidate recall@5 | 100 % |
| Adversarial inputs blocked before the API | 7 / 7 |
| Real AI explanations passing the grounding check | 5 / 5 |
| Real requests that failed because of provider problems (handled, no crash) | 8 / 15 |
| Human review of every real AI response | 4 pass, 2 pass with issues, 0 fail |

---

## 6. Limitations and Biases

- **The catalog leaves out most of the world's music.** I filtered the source dataset to English-language vocal tracks (instrumentals kept) and dropped Latin music entirely. My language filter uses title and artist heuristics plus an exclusion list I reviewed by hand, so it reflects my own judgment about what "reads as English." A listener who wants K-pop, Afrobeats or Spanish-language music gets nothing that fits.
- **Popularity bias.** The catalog keeps the most popular tracks in each genre, so well-known, mostly Western, mainstream artists dominate, and niche artists are almost absent.
- **Uneven genres mean uneven quality.** Most genres have about 70 songs, while reggae has 22, r&b 32 and hip hop 38. Fans of the smaller genres get weaker, less varied lists.
- **Moods are guesses.** Only 19 songs have moods assigned by hand. Every other mood is inferred from audio features, so a song labeled "chill" may not feel chill at all.
- **Weights decide whose taste wins.** With the current weights (genre 1.0, energy 3.0), "calm music for study" returns zero classical songs and "reggae songs for study" returns zero reggae songs, because mood and energy outweigh the genre the user asked for.
- **Taste squeezed into four fields.** The LLM has to reduce any request to one genre, one mood, an energy level and an acoustic yes/no. "Sad lyrics but upbeat tempo," or more than one genre, gets lost.
- **Unused features.** Tempo, danceability and valence are stored but not scored, even though in this catalog they carry information that energy doesn't.
- **Word matching, not meaning.** Retrieval is TF-IDF, so "festival drops" doesn't find the party note unless the words overlap.
- **Confidence measures score, not satisfaction.** The reggae request showed "Strong fit" (82 %) even though no reggae song was returned.

---

## 7. Misuse and Prevention

TuneJudge only recommends songs, so the harm is limited. But it does connect user text to paid LLM APIs, and that creates some risks:

| Misuse | How I prevent or limit it |
|---|---|
| **Prompt injection** ("ignore all previous instructions…", fake `</user_request>` tags) to make the model leak its prompt or ignore its rules | The input guardrail blocks known injection patterns **before any API call** (7 / 7 adversarial inputs blocked). The model also only returns a fixed schema (genre, mood, energy, acoustic), so even a successful injection has very little room to do damage. |
| **Using the app as a free general chatbot** (homework, code, other topics) at my API cost | The parser flags requests that aren't about music (`on_topic = false`) and the app declines them. Requests over 500 characters are rejected. |
| **Making the AI say false things about real songs or artists** | The grounding check rejects any explanation that names a song that wasn't recommended or quotes a number that isn't in the data, and replaces it with the scorer's own reasons. |
| **Leaking API keys** | Keys live in `.env`, which is git-ignored. `.env.example` has placeholders only, and the code treats the placeholder as "no key." |
| **Over-trusting the results** | Every pick shows its point breakdown and a pipeline trace, and the app warns when the genre or mood isn't in the catalog or the best match is weak. |

**What I haven't solved:** there's no rate limiting, so someone could still run up API costs with many valid music requests. The injection filter is pattern-based, so a cleverly reworded attack could get past it. That's acceptable only because the model's output is so constrained. The request log stores raw request text on the local machine. It's git-ignored, but a public deployment would need a privacy notice and a retention limit.

---

## 8. What Surprised Me While Testing Reliability

- **The AI was more honest than my own code.** For "reggae songs for study and calm," my rule-based confidence label said "Strong fit" (82 %), but the LLM's summary said *"none are reggae."* I expected the AI to be the part I had to police. Here the deterministic part was overconfident, and the grounded AI told the truth.
- **The biggest source of failure wasn't the AI's answers. It was reaching the AI at all.** Of 15 real requests, 8 failed because of provider problems: credit limits, 503 overloads and 504 timeouts. Zero failed because the model invented something. One request took **182 seconds** before I added a 30-second timeout. That's why I added multiple providers and auto-switching.
- **"All tests pass" didn't mean "results are good."** All 77 tests passed while a leftover experiment weight was quietly keeping classical music out of study playlists. Tests check that the code does what I wrote. They don't check that what I wrote is what a listener wants. Only reading real outputs caught it.
- **Faking the AI made it testable.** I couldn't make a real API hallucinate or time out on demand. A fake LLM that returns invented song titles or wrong IDs let me prove the grounding fallback works every time, offline, in under a second.

---

## 9. Collaboration with AI

I used Claude Code as a coding and documentation partner: to build the RAG pipeline, guardrails and tests, write the evaluation tooling, and draft the README and this model card from real test output. I treated its output like a teammate's pull request, checking its claims against the code, the test output and the request log.

**A suggestion I accepted: a scripted stand-in LLM for reproducible evidence.** I needed to show that the system works in a way a grader could reproduce, but my API credits had run out. The AI suggested `scripts/guardrail_demo.py`, which runs the full pipeline with a scripted stand-in LLM that is clearly labeled as not real, plus a copy of my real request log as `evidence/ai_requests_log.jsonl`. I accepted it because it produces the same output on every run, needs no API key, and still demonstrates every guardrail, including a hallucinating model being caught and replaced with rule-based reasons.

**A suggestion I didn't accept as written: an example for this reflection.** The AI suggested I use "choosing to keep the experiment weights and document them instead of reverting them" as an example here. When I asked it to explain, it admitted that *it* had made that call while writing the README. I never decided it. I dropped the example, because a reflection has to describe what actually happened, even when a made-up version sounds better. It reminded me to check AI output about *my own work* as carefully as AI-written code.

---

## 10. Ideas for Improvement

1. **Restore a stronger genre weight**, or require at least one song from the requested genre when the catalog has one, so "reggae for studying" returns reggae.
2. **Make confidence reflect the request:** lower the fit level when the requested genre is missing from the results.
3. **Score tempo, danceability and valence**, which carry real information in this catalog.
4. **Partial credit for similar moods** ("happy" ↔ "uplifting"), like the related-genre rule.
5. **Embeddings for retrieval**, so "festival drops" finds the party note.
6. **Rate limiting** and a privacy notice before any public deployment.
7. **A broader catalog**, with non-English and Latin music and better-balanced genres.
