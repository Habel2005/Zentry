# brain.py
import os
import random
import chromadb
from chromadb.utils import embedding_functions
from llama_cpp import Llama
from Phi3.config import DB_PATH, CACHE, safe_guardrails, enforce_short_reply, retrieve_from_cache

class Brain:
    def __init__(self, model_path: str = "phi3-mini.gguf"):
        # --- init DB ---
        self.client = chromadb.PersistentClient(path=DB_PATH)
        self.collection = self.client.get_or_create_collection(
            name="admissions",
            embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
        )
        print(f"[Brain] Collection loaded: {self.collection.name}")

        # --- init LLM (quantized GGUF via llama-cpp) ---
        self.llm = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_threads=os.cpu_count(),
            n_gpu_layers=35  # push layers to GPU if present
        )
        print(f"[Brain] LLM loaded: {model_path}")

    # --- Retrieval helper ---
    def _retrieve_context(self, query: str, top_k: int = 3) -> list[str]:
        results = self.collection.query(query_texts=[query], n_results=top_k)
        docs = results.get("documents", [])
        context_docs = []
        if docs:
            # flatten multiple retrieved chunks into one context block
            for d in docs[0]:
                if d:
                    context_docs.append(d.strip())
        print(f"[Brain] Retrieved {len(context_docs)} context chunks for query: '{query}'")
        return context_docs

    # --- Prompt builder ---
    def _build_prompt(self, query: str, contexts: list[str]) -> str:
        if not contexts:
            return f"User asked: {query}\nAnswer: I don't have that information."
        context_block = "\n".join(contexts)
        return f"""You are Admission Assistant at Toc-H Institute of Science and Technology (TIST).
        Answer factually, politely, and briefly. 
        Use ONLY the context below; if unknown, say 'Not available'.

        Context:
        {context_block}

        Question: {query}
        Answer:"""

    # --- Main ask method ---
    def ask(self, query: str, top_k: int = 3, max_tokens: int = 64) -> str:
        # --- 1) cache check ---
        cached = retrieve_from_cache(query)
        if cached:
            return cached

        # --- 2) retrieve context ---
        context_docs = self._retrieve_context(query, top_k=top_k)
        print(f"[RAG CONTEXT] {context_docs}")

        # --- 3) check for empty context early ---
        if not context_docs:
            reply = "I can only answer based on verified admission info."
            CACHE.set(query, reply)
            return reply

        # --- 4) build prompt ---
        prompt = self._build_prompt(query, context_docs)
        print(f"[PROMPT]\n{prompt}")

        # --- 5) run LLM ---
        out = self.llm(
            prompt,
            max_tokens=max_tokens,
            stop=["User:", "Question:"],
            echo=False
        )
        print(f"RWAAAAAA --->{out}")
        reply = out["choices"][0]["text"].strip()

        # --- 5.5) fast exit if model admits ignorance ---
        if reply.lower().startswith(("not available", "sorry", "i cannot", "no info")):
            fallback_options = [
                "Not available.",
                "I don’t have that info.",
                "That information isn’t provided.",
                "I can only answer from official admission data."
            ]
            reply = random.choice(fallback_options)
            print(f"[FAST EXIT FALLBACK] {reply}")
            CACHE.set(query, reply)
            return reply


        # --- 6) guardrails ---
        print(f"INTO THE SAFE----{reply}")
        reply = safe_guardrails(reply, context_docs)
        print(f"AFTER SAFE ----{reply}")

        # --- 7) enforce brevity ---
        reply = enforce_short_reply(reply, max_words=12)

        # --- 8) cache ---
        CACHE.set(query, reply)

        return reply

    # --- Utility: show current cache contents ---
    def show_cache(self):
        print("\n[Current CACHE]")
        for k, v in CACHE.items():
            print(f"{k}: {v}")
