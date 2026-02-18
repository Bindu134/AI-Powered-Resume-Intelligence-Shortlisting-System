# 🧠 ResumeIQ — AI-Powered Resume Intelligence & Shortlisting System

A production-grade AI resume screening system demonstrating DSA, ML, LLM integration, system design, JWT authentication, and GDPR compliance.

---

## 🏗️ Architecture

```
Frontend (HTML/React)               → localhost:3000
        ↓
Backend (FastAPI + JWT Auth)        → localhost:8000
        ↓
Resume Parser (Anthropic LLM)       → structured extraction
        ↓
Multi-Format Support                → PDF, DOCX, TXT, Images (OCR)
        ↓
Vector Store (sentence-transformers all-mpnet-base-v2 + FAISS) → embeddings
        ↓
Ranking Algorithm                   → cosine sim + skill match + exp scoring
        ↓
Evaluation Metrics                  → precision, recall, F1
        ↓
REST API                            → /api/shortlist
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

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env and add your keys:
# ANTHROPIC_API_KEY=your_key_here
# SECRET_KEY=your-long-random-secret-key
```

> No API key? The system falls back to a rule-based parser automatically.

### 3. Start Backend

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

API docs at: http://localhost:8000/docs

### 4. Get an Auth Token (Required)

```bash
curl -X POST "http://localhost:8000/token" \
  -d "username=admin&password=admin123"
```

Use the returned token in the `Authorization: Bearer <token>` header for all API requests.

### 5. Open Frontend

Open `frontend/index.html` directly in your browser (no build needed).

---

## 🐳 Docker (Production)

```bash
cp .env.example .env   # fill in ANTHROPIC_API_KEY and SECRET_KEY
docker-compose up --build
```

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000

---

## ⚙️ Features

| Feature | Implementation |
|---|---|
| PDF Resume Parsing | PyPDF2 + Anthropic Claude |
| DOCX Resume Parsing | python-docx (tables included) |
| TXT Resume Parsing | UTF-8 / latin-1 fallback |
| Image Resume Parsing | pytesseract OCR (PNG, JPG, JPEG, TIFF) |
| Skill Extraction | LLM prompt engineering |
| Embeddings | sentence-transformers (all-mpnet-base-v2) |
| Vector Search | FAISS IndexFlatIP (cosine ANN) |
| Ranking Algorithm | Weighted scoring (semantic + skill + exp) |
| Evaluation Metrics | Precision, Recall, F1 score |
| JWT Authentication | python-jose + passlib (bcrypt) |
| Role-Based Access | Admin and Recruiter roles |
| GDPR Compliance | Right to erasure endpoints |
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

**Semantic Similarity** — Cosine similarity between job description embedding and resume embedding using `all-mpnet-base-v2` (upgraded from MiniLM for higher accuracy).

**Skill Match Score** — `0.7 × (required_skills_covered) + 0.3 × (preferred_skills_covered)` with fuzzy substring matching.

**Experience Score** — Sigmoid-like function: 1.0 at ≥1.5× required years, scales down below requirement.

**Evaluation Metrics** — Precision, Recall, and F1 score available via `evaluate_ranking()` for labeled datasets.

---

## 🌐 API Endpoints

| Method | Path | Auth | Description |
|---|---|---|---|
| POST | `/token` | Public | Login and get JWT token |
| GET | `/health` | Public | Health check |
| POST | `/api/resume/upload` | Required | Upload single resume (PDF/DOCX/TXT/image) |
| POST | `/api/resume/batch` | Required | Upload up to 20 resumes |
| POST | `/api/job` | Required | Create job description |
| GET | `/api/jobs` | Required | List all jobs |
| POST | `/api/shortlist` | Required | Rank candidates for a job |
| GET | `/api/candidates` | Required | List candidates (filterable) |
| GET | `/api/candidates/{id}` | Required | Get candidate detail |
| DELETE | `/api/candidates/{id}` | Required | Delete candidate |
| DELETE | `/api/gdpr/erase` | Admin only | Erase all data by email (GDPR) |
| GET | `/api/dashboard/stats` | Required | Dashboard stats |

Full interactive docs: http://localhost:8000/docs

---

## 🔐 Authentication

All API endpoints (except `/health` and `/token`) require a valid JWT token.

**Login:**
```bash
POST /token
Body: username=admin&password=admin123
```

**Use token:**
```bash
Authorization: Bearer <your_token>
```

**Default users:**

| Username | Password | Role |
|---|---|---|
| admin | admin123 | Admin |
| recruiter | recruiter123 | Recruiter |

> ⚠️ Change default passwords before deploying to production. Set a strong `SECRET_KEY` in `.env`.

---

## 🛡️ GDPR Compliance

- **Right to Erasure** — Delete a single candidate via `DELETE /api/candidates/{id}`
- **Bulk Erasure by Email** — Admin-only endpoint `DELETE /api/gdpr/erase?email=...` removes all records for a candidate
- All deletions are logged with the acting user's username for audit trails
- Candidate data is also removed from the vector store on deletion

---

## 🧠 Concepts Demonstrated

- **Cosine Similarity** — `vector_store.py` `_cosine()`
- **Vector/ANN Search** — FAISS IndexFlatIP with normalised vectors
- **Ranking Algorithms** — Weighted multi-factor scoring in `ranker.py`
- **Evaluation Metrics** — Precision, Recall, F1 via `evaluate_ranking()` in `ranker.py`
- **Batch Inference** — `asyncio.gather()` for concurrent resume processing
- **API Rate Limiting** — slowapi decorators on upload/shortlist endpoints
- **Prompt Engineering** — Structured JSON extraction prompt in `resume_parser.py`
- **Error Handling & Retries** — Tenacity `@retry` with exponential backoff
- **Explainable AI** — `_explain()` generates per-candidate natural language reasoning
- **JWT Auth** — Stateless token-based authentication with role-based access control
- **OCR** — pytesseract for image-based resume extraction
- **Multi-format Parsing** — PDF, DOCX, TXT, and image support with automatic routing

---

## 🔧 Project Structure

```
resume-ai/
├── backend/
│   ├── main.py            # FastAPI app, routes, JWT auth, GDPR endpoints
│   ├── models.py          # Pydantic schemas
│   ├── resume_parser.py   # PDF/DOCX/TXT/image extraction + LLM parsing
│   ├── vector_store.py    # Embeddings (sentence-transformers + FAISS)
│   ├── ranker.py          # Weighted scoring + explainability + evaluation metrics
│   ├── requirements.txt
│   ├── Dockerfile
│   └── .env.example
├── frontend/
│   └── index.html         # Single-file React app (no build needed)
├── docker-compose.yml
└── README.md
```

---

## 📦 Dependencies

```
# Core
fastapi
anthropic
sentence-transformers       # all-mpnet-base-v2 for embeddings
faiss-cpu
PyPDF2

# Multi-format support
python-docx                 # DOCX parsing
pytesseract                 # OCR for image resumes
Pillow                      # Image handling

# Auth & Security
python-jose[cryptography]   # JWT tokens
passlib[bcrypt]             # Password hashing

# Evaluation
scikit-learn                # Precision, Recall, F1

# Utilities
slowapi                     # Rate limiting
tenacity                    # Retries with exponential backoff
python-dotenv
```

> For OCR support on Windows, also install Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki

---

## 💡 Roadmap

- Bias Detection — audit matched vs. missing skills across demographic signals
- Fine-tuned Skill Classifier — small BERT model for domain-specific skill tagging
- Pinecone Integration — replace FAISS for cloud-native vector search
- Persistent Storage — PostgreSQL + pgvector for production data layer
- CSV/PDF Export — shortlist report generation
- Email Notifications — notify shortlisted candidates automatically
