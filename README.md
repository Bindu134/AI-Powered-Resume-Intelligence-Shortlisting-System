# 🧠 ResumeIQ — AI-Powered Resume Intelligence & Shortlisting System

A production-grade AI resume screening system demonstrating DSA, ML, LLM integration, and system design.

---

## 🏗️ Architecture

```
Frontend (HTML/React)          → localhost:3000
        ↓
Backend (FastAPI)              → localhost:8000
        ↓
Resume Parser (Anthropic LLM)  → structured extraction
        ↓
Vector Store (sentence-transformers + FAISS) → embeddings
        ↓
Ranking Algorithm              → cosine sim + skill match + exp scoring
        ↓
REST API                       → /api/shortlist
```

---

## 🚀 Quick Start

### 1. Clone & Setup

```bash
cd resume-ai/backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure API Key

```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

> **No API key?** The system falls back to a rule-based parser automatically.

### 3. Start Backend

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

API docs at: http://localhost:8000/docs

### 4. Open Frontend

Open `frontend/index.html` directly in your browser (no build needed).

---

## 🐳 Docker (Production)

```bash
cp .env.example .env   # fill in ANTHROPIC_API_KEY
docker-compose up --build
```

Frontend: http://localhost:3000  
Backend API: http://localhost:8000

---

## ⚙️ Features

| Feature | Implementation |
|---------|---------------|
| PDF Resume Parsing | PyPDF2 + Anthropic Claude |
| Skill Extraction | LLM prompt engineering |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector Search | FAISS IndexFlatIP (cosine ANN) |
| Ranking Algorithm | Weighted scoring (semantic + skill + exp) |
| Batch Upload | Async concurrent processing (20 files) |
| Rate Limiting | slowapi (10 req/min uploads, 20 shortlists/min) |
| Error Handling | Tenacity retries (3 attempts, exponential backoff) |
| Explainable AI | Natural language rank explanations |
| Admin Dashboard | Stats, top skills, candidate management |
| Fallback Mode | Rule-based parser + TF-IDF when no API key |

---

## 📐 Scoring Algorithm

```
final_score = 0.50 × semantic_similarity
            + 0.30 × skill_match_score
            + 0.20 × experience_score
```

**Semantic Similarity** — Cosine similarity between job description embedding and resume embedding (sentence-transformers all-MiniLM-L6-v2).

**Skill Match Score** — `0.7 × (required_skills_covered) + 0.3 × (preferred_skills_covered)` with fuzzy substring matching.

**Experience Score** — Sigmoid-like function: 1.0 at ≥1.5× required years, scales down below requirement.

---

## 🌐 API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/resume/upload` | Upload single PDF |
| POST | `/api/resume/batch` | Upload up to 20 PDFs |
| POST | `/api/job` | Create job description |
| GET | `/api/jobs` | List all jobs |
| POST | `/api/shortlist` | Rank candidates for a job |
| GET | `/api/candidates` | List candidates (filterable) |
| GET | `/api/candidates/{id}` | Get candidate detail |
| DELETE | `/api/candidates/{id}` | Delete candidate |
| GET | `/api/dashboard/stats` | Dashboard stats |
| GET | `/health` | Health check |

Full interactive docs: http://localhost:8000/docs

---

## 🧠 Concepts Demonstrated

- **Cosine Similarity** — vector_store.py `_cosine()`
- **Vector/ANN Search** — FAISS IndexFlatIP with normalised vectors
- **Ranking Algorithms** — Weighted multi-factor scoring in ranker.py
- **Batch Inference** — `asyncio.gather()` for concurrent PDF processing
- **API Rate Limiting** — slowapi decorators on upload/shortlist endpoints
- **Prompt Engineering** — Structured JSON extraction prompt in resume_parser.py
- **Error Handling & Retries** — Tenacity `@retry` with exponential backoff
- **Explainable AI** — `_explain()` function generates per-candidate reasoning

---

## 🔧 Project Structure

```
resume-ai/
├── backend/
│   ├── main.py            # FastAPI app, routes, rate limiting
│   ├── models.py          # Pydantic schemas
│   ├── resume_parser.py   # PDF extraction + LLM parsing
│   ├── vector_store.py    # Embeddings (sentence-transformers + FAISS)
│   ├── ranker.py          # Weighted scoring + explainability
│   ├── requirements.txt
│   ├── Dockerfile
│   └── .env.example
├── frontend/
│   └── index.html         # Single-file React app (no build needed)
├── docker-compose.yml
└── README.md
```

---

## 💡 Advanced Extensions (Roadmap)

- **Bias Detection** — audit matched vs. missing skills across demographic signals
- **Fine-tuned Skill Classifier** — small BERT model for domain-specific skill tagging
- **Pinecone Integration** — replace FAISS for cloud-native vector search
- **Persistent Storage** — PostgreSQL + pgvector for production data layer
- **Auth** — JWT-based admin authentication
- **Export** — CSV/PDF shortlist report generation

---

## 📦 Dependencies

Core: `fastapi`, `anthropic`, `sentence-transformers`, `faiss-cpu`, `PyPDF2`  
Utilities: `slowapi` (rate limiting), `tenacity` (retries), `scikit-learn` (TF-IDF fallback)
