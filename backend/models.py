"""Pydantic data models."""
import uuid
from typing import Optional
from pydantic import BaseModel, Field


class JobDescription(BaseModel):
    id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    description: str
    required_skills: list[str] = []
    preferred_skills: list[str] = []
    min_experience_years: int = 0
    created_at: Optional[str] = None


class Project(BaseModel):
    name: str
    description: str
    technologies: list[str] = []


class CandidateProfile(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "Unknown"
    email: Optional[str] = None
    phone: Optional[str] = None
    skills: list[str] = []
    years_experience: float = 0
    education: list[str] = []
    projects: list[Project] = []
    work_history: list[str] = []
    raw_text: Optional[str] = None
    embedding: Optional[list[float]] = None
    filename: Optional[str] = None
    uploaded_at: Optional[str] = None

    def summary_text(self) -> str:
        """Create a text summary for embedding."""
        parts = [
            f"Name: {self.name}",
            f"Skills: {', '.join(self.skills)}",
            f"Experience: {self.years_experience} years",
            f"Education: {'; '.join(self.education)}",
        ]
        if self.projects:
            proj_texts = [f"{p.name}: {p.description}" for p in self.projects[:3]]
            parts.append(f"Projects: {'; '.join(proj_texts)}")
        if self.work_history:
            parts.append(f"Work: {'; '.join(self.work_history[:3])}")
        return " | ".join(parts)


class ShortlistRequest(BaseModel):
    job_id: str
    top_k: int = Field(default=10, ge=1, le=100)
    min_score: float = Field(default=0.0, ge=0.0, le=1.0)


class ScoreBreakdown(BaseModel):
    semantic_similarity: float
    skill_match_score: float
    experience_score: float
    final_score: float


class ShortlistResult(BaseModel):
    rank: int
    candidate: CandidateProfile
    score: float
    score_breakdown: ScoreBreakdown
    matched_skills: list[str]
    missing_skills: list[str]
    explanation: str


class DashboardStats(BaseModel):
    total_candidates: int
    total_jobs: int
    avg_experience: float
    top_skills: dict[str, int]
    candidates_last_24h: int
