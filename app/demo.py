"""Model-free demo/sample fallback for publish preview without GPU/HF downloads."""
from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path

import numpy as np

from app.rag.pipeline import INSUFFICIENT_EVIDENCE
from app.rag.prompt import NO_EVIDENCE_MARKER

logger = logging.getLogger(__name__)

SAMPLE_LABEL = "sample data"
DEMO_EMBED_DIM = 384
SAMPLE_CORPUS_FILES = (
    "data/raw/ornek_rapor.md",
    "experiments/eval/corpus/mavi_korfez_2025.md",
)


def models_available():
    """Return True when heavy model packages import successfully (weights may still be absent)."""
    try:
        import sentence_transformers  # noqa: F401
        import transformers  # noqa: F401
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def resolve_demo_mode(settings):
    """Honor explicit EVIDENCERAG_DEMO; auto enables when model packages are missing."""
    flag = settings.demo_mode
    if flag is True:
        return True
    if flag is False:
        return False
    return not models_available()


def demo_embed(texts, model_name=None):
    """Deterministic bag-of-hashed-tokens vectors; no network or model weights."""
    vectors = []
    for text in texts:
        vec = np.zeros(DEMO_EMBED_DIM, dtype="float32")
        tokens = re.findall(r"\w+", (text or "").casefold())
        if not tokens:
            tokens = ["empty"]
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % DEMO_EMBED_DIM
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[index] += sign
        norm = float(np.linalg.norm(vec)) or 1.0
        vectors.append(vec / norm)
    return np.asarray(vectors, dtype="float32")


def _extract_evidence_texts(user_content):
    """Pull evidence `text` fields out of the serialized prompt user message."""
    import json

    texts = []
    for match in re.finditer(r"Kanıt\s+(\d+)\s*\|\s*(\{.*?\})(?=\n\nKanıt\s+\d+\s*\||\n\nQuestion:|$)", user_content, re.S):
        try:
            payload = json.loads(match.group(2))
        except json.JSONDecodeError:
            continue
        text = payload.get("text")
        if isinstance(text, str) and text.strip():
            texts.append((int(match.group(1)), text.strip()))
    if texts:
        return texts
    # Fallback: any JSON objects with a text field.
    for match in re.finditer(r"\{[^{}]*\"text\"\s*:\s*\".*?\"[^{}]*\}", user_content, re.S):
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        text = payload.get("text")
        if isinstance(text, str) and text.strip():
            texts.append((len(texts) + 1, text.strip()))
    return texts


def demo_generate(messages, model_name=None, max_new_tokens=256):
    """Extract a short answer from evidence text; label is applied by the API layer."""
    del model_name, max_new_tokens
    user_content = ""
    for message in messages or []:
        if message.get("role") == "user":
            user_content = message.get("content") or ""
            break
    if not user_content.strip():
        return NO_EVIDENCE_MARKER
    question = ""
    if "\n\nQuestion:" in user_content:
        question = user_content.split("\n\nQuestion:", 1)[1]
        question = question.split("\n\nAnswer", 1)[0].strip().strip('"')
    evidence = _extract_evidence_texts(user_content)
    if not evidence:
        return NO_EVIDENCE_MARKER
    q_tokens = {t for t in re.findall(r"\w+", question.casefold()) if len(t) > 2}
    best = None
    best_score = -1
    best_index = 1
    for index, text in evidence:
        for sentence in re.split(r"(?<=[.!?。])\s+|\n+", text):
            candidate = sentence.strip()
            if len(candidate) < 8:
                continue
            score = len(q_tokens & set(re.findall(r"\w+", candidate.casefold())))
            if score > best_score:
                best_score = score
                best = candidate
                best_index = index
    if not best:
        best_index, best = evidence[0]
    if "[Kanıt" not in best and "[Evidence" not in best:
        best = f"{best.rstrip()} [Kanıt {best_index}]"
    return best[:800]

def sample_corpus_paths(project_directory: Path):
    paths = []
    for relative in SAMPLE_CORPUS_FILES:
        path = project_directory / relative
        if path.is_file():
            paths.append(path)
    return paths


def seed_sample_corpus(pipeline, project_directory: Path):
    """Index bundled fixture documents with demo embeddings when the store is empty."""
    if pipeline.vector_store.chunks:
        logger.info("Demo mode: existing index retained (%s chunks)", len(pipeline.vector_store.chunks))
        return {"seeded": 0, "documents": []}
    seeded = []
    for path in sample_corpus_paths(project_directory):
        try:
            result = pipeline.index_file(path, filename=path.name)
            seeded.append(result)
        except Exception as exc:
            logger.warning("Demo seed skipped for %s: %s", path.name, exc)
    logger.info("Demo mode: seeded %s sample document(s)", len(seeded))
    return {"seeded": len(seeded), "documents": seeded}


def mark_sample_payload(payload: dict):
    """Attach an honest sample-data marker for UI/API consumers."""
    if not isinstance(payload, dict):
        return payload
    marked = dict(payload)
    marked["sample_data"] = True
    answer = marked.get("answer")
    if isinstance(answer, str) and answer and SAMPLE_LABEL not in answer.casefold():
        if answer == INSUFFICIENT_EVIDENCE:
            marked["answer"] = f"{answer} ({SAMPLE_LABEL})"
        else:
            marked["answer"] = f"[{SAMPLE_LABEL}] {answer}"
    return marked
