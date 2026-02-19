"""
Candidate Ranker
Scoring formula:
  final = w1 * semantic_similarity + w2 * skill_match + w3 * experience_score

Weights (configurable):
  w1=0.50  semantic similarity (embedding cosine)
  w2=0.30  skill match ratio (required skills covered)
  w3=0.20  experience score (normalised years vs requirement)
"""

import numpy as np
import logging
from sentence_transformers import SentenceTransformer
from sklearn.metrics import precision_score, recall_score

from models import CandidateProfile, JobDescription, ShortlistResult, ScoreBreakdown

logger = logging.getLogger(__name__)

# Improved embedding model
model = SentenceTransformer('all-mpnet-base-v2')

WEIGHTS = {"semantic": 0.50, "skill": 0.30, "experience": 0.20}


def _cosine(a: list[float], b: np.ndarray) -> float:
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


def _skill_match(candidate_skills: list[str], required: list[str], preferred: list[str]) -> tuple[float, list[str], list[str]]:
    """Return (score, matched_required, missing_required)."""
    if not required and not preferred:
        return 0.5, [], []

    cand_lower = {s.lower() for s in candidate_skills}

    def fuzzy_match(skill: str) -> bool:
        sl = skill.lower()
        for cs in cand_lower:
            if sl in cs or cs in sl:
                return True
        return False

    matched_req = [s for s in required if fuzzy_match(s)]
    missing_req = [s for s in required if not fuzzy_match(s)]
    matched_pref = [s for s in preferred if fuzzy_match(s)]

    req_score = len(matched_req) / len(required) if required else 1.0
    pref_score = len(matched_pref) / len(preferred) if preferred else 0.5
    score = 0.7 * req_score + 0.3 * pref_score

    all_matched = matched_req + matched_pref
    return float(score), all_matched, missing_req


def _experience_score(candidate_years: float, required_years: int) -> float:
    """Sigmoid-like score: peaks at required_years, slight boost for more."""
    if required_years == 0:
        return min(1.0, candidate_years / 3.0) if candidate_years < 3 else 1.0
    ratio = candidate_years / required_years
    if ratio >= 1.5:
        return 1.0
    elif ratio >= 1.0:
        return 0.9 + 0.1 * (ratio - 1.0) / 0.5
    elif ratio >= 0.7:
        return 0.6 + 0.3 * (ratio - 0.7) / 0.3
    else:
        return max(0.0, ratio / 0.7 * 0.6)


def _explain(rank: int, score: float, breakdown: ScoreBreakdown, matched: list[str], missing: list[str], candidate_name: str, job_title: str) -> str:
    """Generate a natural language explanation."""
    lines = [f"{candidate_name} is ranked #{rank} for {job_title} with an overall score of {score:.0%}."]

    if breakdown.semantic_similarity >= 0.7:
        lines.append("The resume strongly aligns with the job description semantically.")
    elif breakdown.semantic_similarity >= 0.5:
        lines.append("The resume shows moderate semantic alignment with the job description.")
    else:
        lines.append("Semantic alignment with the job description is limited.")

    if matched:
        lines.append(f"Matched skills: {', '.join(matched[:6])}{'...' if len(matched) > 6 else ''}.")
    if missing:
        lines.append(f"Missing required skills: {', '.join(missing[:4])}{'...' if len(missing) > 4 else ''}.")

    if breakdown.experience_score >= 0.9:
        lines.append("Experience level meets or exceeds requirements.")
    elif breakdown.experience_score >= 0.6:
        lines.append("Experience level is close to the requirement.")
    else:
        lines.append("Experience level is below the stated requirement.")

    return " ".join(lines)


def evaluate_ranking(y_true: list[int], y_pred: list[int]) -> dict:
    """
    Evaluate ranking quality using precision, recall, and F1.
    y_true: list of 1s and 0s (1 = good fit, 0 = bad fit) - ground truth
    y_pred: list of 1s and 0s - model predictions
    """
    if not y_true or not y_pred:
        logger.warning("Empty labels provided for evaluation.")
        return {}

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    metrics = {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
    }
    logger.info(f"Evaluation Metrics: {metrics}")
    return metrics


def embed_text(text: str) -> list[float]:
    """Generate embedding for any text using the improved model."""
    return model.encode(text, normalize_embeddings=True).tolist()


class CandidateRanker:
    def rank(
        self,
        job: JobDescription,
        job_embedding: np.ndarray,
        candidates: list[CandidateProfile],
        top_k: int = 10,
        min_score: float = 0.0,
    ) -> list[ShortlistResult]:
        results: list[ShortlistResult] = []

        for candidate in candidates:
            # 1. Semantic similarity
            if candidate.embedding:
                sim = _cosine(candidate.embedding, job_embedding)
            else:
                logger.warning(f"No embedding for {candidate.name}, generating on the fly.")
                generated_embedding = embed_text(candidate.raw_text if hasattr(candidate, 'raw_text') else " ".join(candidate.skills))
                sim = _cosine(generated_embedding, job_embedding)

            # 2. Skill match
            skill_score, matched, missing = _skill_match(
                candidate.skills, job.required_skills, job.preferred_skills
            )

            # 3. Experience
            exp_score = _experience_score(candidate.years_experience, job.min_experience_years)

            # 4. Weighted final
            final = (
                WEIGHTS["semantic"] * sim
                + WEIGHTS["skill"] * skill_score
                + WEIGHTS["experience"] * exp_score
            )

            if final < min_score:
                continue

            breakdown = ScoreBreakdown(
                semantic_similarity=round(sim, 4),
                skill_match_score=round(skill_score, 4),
                experience_score=round(exp_score, 4),
                final_score=round(final, 4),
            )

            results.append(ShortlistResult(
                rank=0,
                candidate=candidate,
                score=round(final, 4),
                score_breakdown=breakdown,
                matched_skills=matched,
                missing_skills=missing,
                explanation="",
            ))

        # Sort descending
        results.sort(key=lambda r: r.score, reverse=True)
        results = results[:top_k]

        for i, r in enumerate(results, 1):
            r.rank = i
            r.explanation = _explain(
                i, r.score, r.score_breakdown,
                r.matched_skills, r.missing_skills,
                r.candidate.name, job.title
            )

        logger.info(f"Ranked {len(results)} candidates for job '{job.title}'")
        return results