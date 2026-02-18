"""
AI-Powered Resume Intelligence & Shortlisting System
FastAPI Backend
"""

import os
import json
import uuid
import time
import asyncio
import logging
from typing import Optional
from datetime import datetime

import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv

load_dotenv()

from resume_parser import ResumeParser
from vector_store import VectorStore
from ranker import CandidateRanker
from models import (
    JobDescription, CandidateProfile, ShortlistRequest,
    ShortlistResult, DashboardStats
)

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Rate Limiter ──────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Resume Intelligence API",
    description="AI-powered resume screening & shortlisting",
    version="1.0.0",
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory stores (production would use a real DB) ─────────────────────────
candidates_db: dict[str, CandidateProfile] = {}
jobs_db: dict[str, JobDescription] = {}

# ── Services ──────────────────────────────────────────────────────────────────
parser = ResumeParser()
vector_store = VectorStore()
ranker = CandidateRanker()


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


# ── Resume Upload & Parse ─────────────────────────────────────────────────────
@app.post("/api/resume/upload", response_model=CandidateProfile)
@limiter.limit("10/minute")
async def upload_resume(request: Request, file: UploadFile = File(...)):
    """Upload a PDF resume → extract skills/exp/projects → store embedding."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")

    content = await file.read()
    if len(content) > 5 * 1024 * 1024:
        raise HTTPException(413, "File size must be < 5 MB")

    try:
        candidate = await parser.parse(content, file.filename)
        embedding = vector_store.embed(candidate.summary_text())
        candidate.embedding = embedding.tolist()
        candidates_db[candidate.id] = candidate
        logger.info(f"Parsed candidate {candidate.id}: {candidate.name}")
        return candidate
    except Exception as e:
        logger.error(f"Parse error: {e}")
        raise HTTPException(500, f"Failed to parse resume: {str(e)}")


# ── Batch Upload ──────────────────────────────────────────────────────────────
@app.post("/api/resume/batch", response_model=list[CandidateProfile])
@limiter.limit("3/minute")
async def batch_upload(request: Request, files: list[UploadFile] = File(...)):
    """Upload multiple resumes concurrently (max 20)."""
    if len(files) > 20:
        raise HTTPException(400, "Maximum 20 files per batch")

    async def process_one(f: UploadFile):
        try:
            content = await f.read()
            c = await parser.parse(content, f.filename)
            emb = vector_store.embed(c.summary_text())
            c.embedding = emb.tolist()
            candidates_db[c.id] = c
            return c
        except Exception as e:
            logger.warning(f"Skipping {f.filename}: {e}")
            return None

    results = await asyncio.gather(*[process_one(f) for f in files])
    return [r for r in results if r is not None]


# ── Job Description ───────────────────────────────────────────────────────────
@app.post("/api/job", response_model=JobDescription)
async def create_job(job: JobDescription):
    """Register a job description."""
    job.id = job.id or str(uuid.uuid4())
    jobs_db[job.id] = job
    return job


@app.get("/api/jobs", response_model=list[JobDescription])
async def list_jobs():
    return list(jobs_db.values())


# ── Shortlist ─────────────────────────────────────────────────────────────────
@app.post("/api/shortlist", response_model=list[ShortlistResult])
@limiter.limit("20/minute")
async def shortlist_candidates(request: Request, req: ShortlistRequest):
    """
    Rank all candidates for a given job description.
    Scoring = 0.5 * cosine_similarity + 0.3 * skill_match + 0.2 * experience_score
    """
    if not candidates_db:
        raise HTTPException(404, "No candidates in the system")

    job = jobs_db.get(req.job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    job_emb = vector_store.embed(job.description)
    results = ranker.rank(
        job=job,
        job_embedding=job_emb,
        candidates=list(candidates_db.values()),
        top_k=req.top_k,
        min_score=req.min_score,
    )
    return results


# ── Candidate CRUD ────────────────────────────────────────────────────────────
@app.get("/api/candidates", response_model=list[CandidateProfile])
async def list_candidates(skill: Optional[str] = None, min_exp: Optional[int] = None):
    """List candidates with optional filters."""
    candidates = list(candidates_db.values())
    if skill:
        candidates = [c for c in candidates if skill.lower() in [s.lower() for s in c.skills]]
    if min_exp is not None:
        candidates = [c for c in candidates if c.years_experience >= min_exp]
    return candidates


@app.get("/api/candidates/{cid}", response_model=CandidateProfile)
async def get_candidate(cid: str):
    c = candidates_db.get(cid)
    if not c:
        raise HTTPException(404, "Candidate not found")
    return c


@app.delete("/api/candidates/{cid}")
async def delete_candidate(cid: str):
    if cid not in candidates_db:
        raise HTTPException(404, "Candidate not found")
    del candidates_db[cid]
    return {"deleted": cid}


# ── Dashboard Stats ───────────────────────────────────────────────────────────
@app.get("/api/dashboard/stats", response_model=DashboardStats)
async def dashboard_stats():
    all_skills: list[str] = []
    total_exp = 0
    for c in candidates_db.values():
        all_skills.extend(c.skills)
        total_exp += c.years_experience

    from collections import Counter
    skill_counts = Counter(all_skills)
    n = len(candidates_db)

    return DashboardStats(
        total_candidates=n,
        total_jobs=len(jobs_db),
        avg_experience=round(total_exp / n, 1) if n else 0,
        top_skills=dict(skill_counts.most_common(10)),
        candidates_last_24h=n,  # simplified for demo
    )
