# config.py
import os
import time
import hashlib
from sentence_transformers import SentenceTransformer, util
import string
# ----------------
# Paths & Models
# ----------------
script_dir = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(script_dir, "db")
DATA_PATH = os.path.join(script_dir, "data")
DEVICE = "cuda"

# ----------------
# Cache
# ----------------
class SimpleCache:
    def __init__(self, ttl: int = 600):
        self.ttl = ttl
        self.store = {}

    def _make_key(self, query: str) -> str:
        return hashlib.sha256(query.encode()).hexdigest()

    def get(self, query: str):
        k = self._make_key(query)
        if k in self.store:
            val, ts = self.store[k]
            if time.time() - ts < self.ttl:
                return val
            else:
                del self.store[k]
        return None

    def set(self, query: str, value: str):
        k = self._make_key(query)
        self.store[k] = (value, time.time())

    def items(self):
        """Return valid cache items as (query_hash, value)"""
        # filter expired items
        current_time = time.time()
        valid_items = {}
        for k, (v, ts) in self.store.items():
            if current_time - ts < self.ttl:
                valid_items[k] = v
        return valid_items.items()

def retrieve_from_cache(query: str):
    """
    Utility to inspect current cache content for a query.
    Returns stored reply or None.
    """
    cached = CACHE.get(query)
    if cached:
        print(f"[CACHE HIT] {query} -> {cached}")
    else:
        print(f"[CACHE MISS] {query}")
    return cached

CACHE = SimpleCache(ttl=600)

# ----------------
# Embedder for semantic checks
# ----------------
EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
KEYWORD_THRESHOLD = 0.3     # % of query words in context to accept short answer
SEMANTIC_THRESHOLD = 0.75   # good enough semantic match
FALLBACK_THRESHOLD = 0.55   # lower bar for "fast acceptance"

def semantic_guardrails(reply, context_docs, threshold=SEMANTIC_THRESHOLD):
    if not context_docs:
        return 0.0  # no context means no similarity
    emb_reply = EMBEDDER.encode(reply, convert_to_tensor=True)
    sims = []
    for doc in context_docs:
        emb_doc = EMBEDDER.encode(doc, convert_to_tensor=True)
        sims.append(util.cos_sim(emb_reply, emb_doc).item())
    return max(sims) if sims else 0.0


def safe_guardrails(reply, context_docs):
    # 1. If model already admits not knowing → trust it instantly.
    if "not available" in reply.lower():
        return "Not available."

    # 2. Quick semantic check
    score = semantic_guardrails(reply, context_docs)

    # Fast path: if above FALLBACK_THRESHOLD → accept as-is
    if score >= FALLBACK_THRESHOLD:
        # But only *promote* to final if it passes strong threshold
        if score >= SEMANTIC_THRESHOLD:
            return reply  # confident
        else:
            # weaker match: trim to safe short form
            return reply.split(".")[0]  # only first sentence (avoids drift)

    # 3. If score too low → reject
    return "Not available."


def enforce_short_reply(reply: str, max_words: int = 6) -> str:
    """Truncate smartly without cutting mid-word or dropping key tokens."""
    words = reply.strip().split()
    if not words:
        return "No info available."

    # Take up to N words
    short = words[:max_words]

    # Strip trailing punctuation instead of deleting the word
    short[-1] = short[-1].rstrip(string.punctuation)

    return " ".join(short)