"""
Vector Store - sentence-transformers embeddings + FAISS ANN search.
Falls back to TF-IDF cosine similarity if transformers unavailable.
"""
import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    USE_TRANSFORMERS = True
    logger.info("sentence-transformers available, will load model on first use")
except Exception as e:
    USE_TRANSFORMERS = False
    logger.warning(f"sentence-transformers unavailable ({e}). Using TF-IDF fallback.")

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    logger.warning("faiss-cpu unavailable. Brute-force cosine search will be used.")

_model = None

def _get_model():
    global _model
    if _model is None and USE_TRANSFORMERS:
        logger.info("Loading sentence-transformers model...")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


class TFIDFEmbedder:
    """Lightweight TF-IDF based text vectorizer as fallback."""
    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(max_features=512, stop_words="english")
        self._fitted = False
        self._corpus = []

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        self._corpus = texts
        self._fitted = True
        return self.vectorizer.fit_transform(texts).toarray().astype(np.float32)

    def transform(self, text: str) -> np.ndarray:
        if not self._fitted:
            return self.vectorizer.fit_transform([text]).toarray()[0].astype(np.float32)
        return self.vectorizer.transform([text]).toarray()[0].astype(np.float32)


_tfidf = TFIDFEmbedder()


class VectorStore:
    def __init__(self, dim: int = 384):
        self.dim = dim
        self.index = None
        self.id_map: list[str] = []
        if HAS_FAISS:
            self.index = faiss.IndexFlatIP(dim)

    def embed(self, text: str) -> np.ndarray:
        """Convert text to normalised embedding vector."""
        if USE_TRANSFORMERS:
            vec = _get_model().encode([text], normalize_embeddings=True, show_progress_bar=False)[0]
            return vec.astype(np.float32)
        else:
            vec = _tfidf.transform(text)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            if len(vec) < self.dim:
                vec = np.pad(vec, (0, self.dim - len(vec)))
            else:
                vec = vec[:self.dim]
            return vec.astype(np.float32)

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two unit vectors."""
        dot = float(np.dot(a, b))
        norm_a = float(np.linalg.norm(a))
        norm_b = float(np.linalg.norm(b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def delete(self, candidate_id: str) -> None:
        """Remove a candidate from the id_map."""
        if candidate_id in self.id_map:
            self.id_map.remove(candidate_id)
            logger.info(f"Candidate {candidate_id} removed from vector store.")
        else:
            logger.warning(f"Candidate {candidate_id} not found in vector store.")