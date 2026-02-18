"""
AI-Powered Resume Intelligence & Shortlisting System
FastAPI Backend — with JWT Auth & GDPR-style data privacy
"""

import os
import json
import uuid
import time
import asyncio
import logging
from typing import Optional
from datetime import datetime, timedelta

import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv
from jose import JWTError, jwt
from passlib.context import CryptContext

load_dotenv()

from resume_parser import ResumeParser
from vector_store import VectorStore
from ranker import CandidateRanker
from models import (
    JobDescription, CandidateProfile, ShortlistRequest,
    ShortlistResult, DashboardStats
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Security Config ───────────────────────────────────────────────────────────
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# ── Fake Users DB (replace with real DB in production) ────────────────────────
USERS_DB = {
    "admin": {
        "username": "admin",
        "hashed_password": pwd_context.hash("admin123"),  # change in production
        "role": "admin",
    },
    "recruiter": {
        "username": "recruiter",
        "hashed_password": pwd_context.hash("recruiter123"),  # change in production
        "role": "recruiter",
    },
}

# ── Rate Limiter ──────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Resume Intelligence API",
    description="AI-powered resume screening & shortlisting",
    version="2.0.0",
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


# ── Auth Helpers ──────────────────────────────────────────────────────────────

class Token(BaseModel):
    access_token: str
    token_type: str


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def create_access_token(data: dict, expires_delta: timedelta = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


async def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = USERS_DB.get(username)
    if user is None:
        raise credentials_exception
    return user


async def require_admin(current_user: dict = Depends(get_current_user)):
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user


# ── Auth Endpoints ────────────────────────────────────────────────────────────

@app.post("/token", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Login to get a JWT access token."""
    user = USERS_DB.get(form_data.username)
    if not user or not verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = create_access_token(data={"sub": user["username"], "role": user["role"]})
    logger.info(f"User '{user['username']}' logged in successfully.")
    return {"access_token": token, "token_type": "bearer"}


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


# ── Resume Upload & Parse ─────────────────────────────────────────────────────

ALLOWED_EXTENSIONS = (".pdf", ".docx", ".txt", ".png", ".jpg", ".jpeg")

@app.post("/api/resume/upload", response_model=CandidateProfile)
@limiter.limit("10/minute")
async def upload_resume(
    request: Request,
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
):
    """Upload a resume (PDF/DOCX/TXT/image) → extract skills/exp/projects → store embedding."""
    if not any(file.filename.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS):
        raise HTTPException(400, f"Supported formats: {', '.join(ALLOWED_EXTENSIONS)}")

    content = await file.read()
    if len(content) > 5 * 1024 * 1024:
        raise HTTPException(413, "File size must be < 5 MB")

    try:
        candidate = await parser.parse(content, file.filename)
        embedding = vector_store.embed(candidate.summary_text())
        candidate.embedding = embedding.tolist()
        candidates_db[candidate.id] = candidate
        logger.info(f"Parsed candidate {candidate.id}: {candidate.name} by user '{current_user['username']}'")
        return
