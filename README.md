# 🔮 MyNaksh — Astro Conversational Insight Agent

A multi-turn conversational AI service that provides personalized astrological guidance using **RAG (Retrieval-Augmented Generation)** over a curated Vedic & Western astrology knowledge base, powered by **Google Gemini 2.5 Flash**.

## Features

- **Multi-turn Conversations** — Session-based memory with sliding window (last 10 turns)
- **Interactive CLI** — Terminal-based conversation loop with "continue? y/n" checkpoints
- **Intent-Aware RAG** — Retrieves knowledge only when it adds value; skips for greetings, summaries, meta-questions
- **Personalization** — Derives zodiac sign from birth date, tailors all responses
- **Hindi Toggle** — Supports Hindi (Devanagari) responses via `preferred_language: "hi"`
- **FAISS Vector Search** — Semantic embeddings with `sentence-transformers` (`all-MiniLM-L6-v2`) + FAISS index
- **LLM Abstraction** — Google Gemini 2.5 Flash or Stub mode (works without API key)

## Architecture

```
┌─────────────────────────────────────────────────────┐
│              FastAPI /chat  &  CLI                   │
├──────────┬──────────┬──────────────┬────────────────┤
│ Session  │ Profile  │    RAG       │    LLM         │
│ Manager  │ (Zodiac) │  Pipeline    │   Client       │
│          │          │              │                │
│ Memory   │ DOB →    │ Intent →     │ Gemini 2.5     │
│ Window   │ Zodiac   │ FAISS →     │ Flash /        │
│ (last N) │ Sign     │ Retrieve    │ Stub           │
└──────────┴──────────┴──────────────┴────────────────┘
                          ↕
                   data/ Knowledge Base
```

## Project Structure

```
MyNaksh/
├── cli.py                      # Interactive multi-turn CLI
├── requirements.txt            # Python dependencies
├── .env                        # Gemini API key (user-provided)
├── data/
│   ├── zodiac_traits.json      # 12 zodiac signs
│   ├── planetary_impacts.json  # 9 planets (malefic/benefic)
│   ├── career_guidance.txt     # Career advice entries
│   ├── love_guidance.txt       # Love/relationship entries
│   ├── spiritual_guidance.txt  # Spiritual guidance entries
│   └── nakshatra_mapping.json  # 27 nakshatras
├── app/
│   ├── main.py                 # FastAPI app + /chat endpoint
│   ├── models.py               # Pydantic request/response models
│   ├── profile.py              # Zodiac derivation from DOB
│   ├── session.py              # Session-based conversation memory
│   ├── rag/
│   │   ├── indexer.py          # FAISS index builder
│   │   ├── retriever.py        # Semantic search retriever
│   │   └── intent.py           # Intent-aware retrieval logic
│   └── llm/
│       └── client.py           # Gemini / Stub LLM abstraction
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Gemini API Key

Create a `.env` file (or edit the existing one) with your Google Gemini API key:

```
GEMINI_API_KEY=your-gemini-api-key-here
```

Without an API key, the app runs in **stub mode** with template-based responses.

### 3. Run the Interactive CLI (Recommended)

```bash
python3 cli.py
```

This starts an interactive session:
1. Enter your details (name, DOB, birth time, place, language)
2. Your zodiac sign is derived automatically
3. Ask questions — the agent responds using RAG + Gemini
4. After each answer: **"Want to continue? (y/n)"**
5. Type `y` to keep going, `n` to exit

**Example session:**
```
🔮 Initializing Astro Agent...
✅ Ready! Indexed 108 knowledge chunks.

=======================================================
  🌟 Astro Conversational Insight Agent 🌟
=======================================================

Let me know a bit about you first:

  Your name: Priyansh
  Date of birth (YYYY-MM-DD): 1999-09-12
  Time of birth (HH:MM): 13:09
  Place of birth: Indore, India
  Preferred language (en/hi) [en]: en

  ✨ Your zodiac sign: Virgo
  🎂 Age: 26

[Turn 1]
You: How will my 2026 be?

🔭 Thinking...

🌟 Astro Guide (Virgo):
  As a Virgo, 2026 brings a period of growth and transformation...

  📚 Sources: zodiac_traits, career_guidance
  🔍 Retrieval used: Yes

Want to continue? (y/n): y

[Turn 2]
You: What about my love life?
...
```

### 4. Run the API Server (Alternative)

```bash
uvicorn app.main:app --reload --port 8000
```

### 5. Test the API

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "test-001",
    "message": "What should I focus on in my career this month?",
    "user_profile": {
      "name": "Ritika",
      "birth_date": "1995-08-20",
      "birth_time": "14:30",
      "birth_place": "Jaipur, India",
      "preferred_language": "en"
    }
  }'
```

### 6. Hindi Response

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "test-002",
    "message": "मेरे करियर में इस महीने क्या होगा?",
    "user_profile": {
      "name": "Ritika",
      "birth_date": "1995-08-20",
      "preferred_language": "hi"
    }
  }'
```

## API Contract

### POST `/chat`

**Request:**
```json
{
  "session_id": "abc-123",
  "message": "How will my month be in career?",
  "user_profile": {
    "name": "Ritika",
    "birth_date": "1995-08-20",
    "birth_time": "14:30",
    "birth_place": "Jaipur, India",
    "preferred_language": "hi"
  }
}
```

**Response:**
```json
{
  "response": "आपके लिए यह महीना अवसर लेकर आ रहा है...",
  "zodiac": "Leo",
  "context_used": ["career_guidance", "zodiac_traits"],
  "retrieval_used": true
}
```

### GET `/health`

Returns index size and LLM mode status.

## Knowledge Base

| File | Description |
|------|-------------|
| `zodiac_traits.json` | All 12 zodiac signs — personality, strengths, challenges |
| `planetary_impacts.json` | 9 planets — malefic/benefic, challenges, opportunities |
| `career_guidance.txt` | 20 career-related guidance entries |
| `love_guidance.txt` | 20 relationship/love guidance entries |
| `spiritual_guidance.txt` | 20 spiritual growth guidance entries |
| `nakshatra_mapping.json` | All 27 nakshatras with ruling planets, deities, and qualities |

## Quality & Cost Awareness

### Retrieval Helps (Example)
> **Query:** "What career opportunities does Leo have?"
> **Result:** The system retrieves Leo traits + career guidance → produces a grounded, specific response.

### Retrieval Hurts (Example)
> **Query:** "Summarize what you've told me so far"
> **Result:** Retrieval would add irrelevant zodiac/career chunks. The intent classifier correctly skips retrieval and serves from session memory.

## Tech Stack

| Component | Technology |
|-----------|------------|
| LLM | Google Gemini 2.5 Flash |
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2) |
| Vector Search | FAISS (faiss-cpu) |
| API Framework | FastAPI + Uvicorn |
| Data Validation | Pydantic |
| Environment | python-dotenv |
