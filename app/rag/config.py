import os
import json
import re
from functools import lru_cache
import ipaddress
from pathlib import Path
from dataclasses import dataclass, field
from urllib.parse import urlsplit

from app.access import normalize_profile, parse_public_hosts

XAI_DEFAULT_MODEL = "grok-3-mini"
XAI_DEFAULT_API_URL = "https://api.x.ai/v1/chat/completions"
XAI_MODEL_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$")
_ALLOWED_XAI_HOST_SUFFIX = ".x.ai"


class InputLimitError(ValueError):
    """An explicit input or store budget was exceeded (not a server failure)."""


def normalize_llm_provider(value):
    """Accept 'local' or 'xai' (alias 'grok')."""
    provider = (value or "local").strip().lower()
    if provider == "grok":
        return "xai"
    if provider not in {"local", "xai"}:
        raise ValueError("LLM_PROVIDER must be 'local' or 'xai' (alias 'grok')")
    return provider


LLM_TORCH_DTYPES = frozenset({"auto", "float32", "bfloat16", "float16"})


def normalize_llm_torch_dtype(value):
    """Accept 'auto', 'float32', 'bfloat16', or 'float16'."""
    if value is None:
        return "auto"
    if not isinstance(value, str):
        raise ValueError("LLM_TORCH_DTYPE must be one of: auto, float32, bfloat16, float16")
    dtype = value.strip().lower() or "auto"
    if dtype not in LLM_TORCH_DTYPES:
        raise ValueError("LLM_TORCH_DTYPE must be one of: auto, float32, bfloat16, float16")
    return dtype


def validate_xai_model_name(model_name):
    """Validate xAI Grok model identifiers strictly to prevent path/payload injection."""
    if not isinstance(model_name, str) or not XAI_MODEL_NAME_RE.fullmatch(model_name):
        raise ValueError(
            "Invalid xAI model name: must match ^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$"
        )
    return model_name


def validate_xai_api_url(url):
    """Validate xAI endpoint URL strictly (HTTPS, x.ai domain only, no SSRF targets)."""
    if not isinstance(url, str) or not url.strip() or any(c.isspace() or ord(c) < 32 for c in url):
        raise ValueError("Invalid xAI API URL")
    parsed = urlsplit(url)
    if parsed.scheme != "https":
        raise ValueError("xAI API URL must use https")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("xAI API URL must not contain userinfo credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("xAI API URL must not contain query parameters or fragments")
    host = (parsed.hostname or "").lower()
    if not host:
        raise ValueError("xAI API URL must include a hostname")
    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        raise ValueError("xAI API URL must not target a raw IP address")
    if host != "api.x.ai" and not host.endswith(_ALLOWED_XAI_HOST_SUFFIX):
        raise ValueError("xAI API URL host must be api.x.ai or a *.x.ai host")
    if parsed.port not in (None, 443):
        raise ValueError("xAI API URL must use port 443")
    path = parsed.path or ""
    if not path.startswith("/v1/") or "\\" in path or "/../" in path or path.endswith("/.."):
        raise ValueError("xAI API URL path must stay under /v1/")
    return url


@lru_cache(maxsize=1)
def model_manifest():
    return json.loads((Path(__file__).resolve().parents[2] / "models.lock.json").read_text(encoding="utf-8"))["models"]


def effective_revision(kind, model_name, revision=None):
    if revision is None:
        settings = Settings()
        expected_name = settings.embedding_model_name if kind == "embedding" else settings.llm_model_name
        if model_name != expected_name:
            raise ValueError("Explicit immutable revision required for a different model repository")
        revision = settings.embedding_model_revision if kind == "embedding" else settings.llm_model_revision
    if not isinstance(revision,str) or not re.fullmatch(r"[0-9a-f]{40}",revision):
        raise ValueError("Model revision must be an immutable 40-character commit SHA")
    return revision


def _str_env(name, default):
    return lambda: os.getenv(name) or default


def _first_env(names, default):
    def read():
        for name in names:
            value = os.getenv(name)
            if value:
                return value
        return default

    return read


def _prefix_env(name, default):
    # Unlike other settings, an empty value is meaningful here: it disables the prefix.
    return lambda: os.getenv(name, default)


def _int_env(name, default):
    def read():
        value = os.getenv(name)
        try:
            return int(value) if value else default
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer") from exc

    return read


def _bool_factory(name, default):
    def read():
        value = os.getenv(name)
        if not value:
            return default
        if value.lower() in ("1", "true", "yes", "on"):
            return True
        if value.lower() in ("0", "false", "no", "off"):
            return False
        raise ValueError(f"{name} must be true or false, got {value!r}")

    return read


def _float_env(name, default):
    def read():
        value = os.getenv(name)
        return float(value) if value else default

    return read


def _bool_env(name):
    value = os.getenv(name, "false").lower()
    if value not in {"true", "false", "1", "0"}:
        raise ValueError(f"{name} must be true/false or 1/0")
    return value in {"true", "1"}


# Environment variables are read when Settings() is created, not when this module is
# imported, so values loaded from a .env file at startup are picked up.
def _demo_env():
    """Tri-state demo flag: unset/false off; true on; auto when heavy model deps are missing."""
    value = os.getenv("EVIDENCERAG_DEMO")
    if value is None or value.strip() == "":
        return False
    lowered = value.strip().lower()
    if lowered == "auto":
        return None
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise ValueError("EVIDENCERAG_DEMO must be true/false/auto")


@dataclass(frozen=True)
class Settings:
    access_profile: str = field(default_factory=_str_env("ACCESS_PROFILE", "local"))
    api_key: str | None = field(default_factory=_str_env("EVIDENCERAG_API_KEY", None), repr=False)
    public_hosts: tuple = field(default_factory=lambda: parse_public_hosts(os.getenv("EVIDENCERAG_PUBLIC_HOST")))
    demo_mode: bool | None = field(default_factory=_demo_env)
    embedding_model_name: str = field(
        default_factory=_str_env("EMBEDDING_MODEL_NAME", "intfloat/multilingual-e5-small")
    )
    # E5 models expect these prefixes; set them to an empty string for models that don't.
    embedding_query_prefix: str = field(default_factory=_prefix_env("EMBEDDING_QUERY_PREFIX", "query: "))
    embedding_passage_prefix: str = field(
        default_factory=_prefix_env("EMBEDDING_PASSAGE_PREFIX", "passage: ")
    )
    llm_provider: str = field(default_factory=_str_env("LLM_PROVIDER", "local"))
    llm_model_name: str = field(default_factory=_str_env("LLM_MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct"))
    llm_torch_dtype: str = field(default_factory=_str_env("LLM_TORCH_DTYPE", "auto"))
    embedding_model_revision: str | None = field(default_factory=_str_env("EMBEDDING_MODEL_REVISION", None))
    llm_model_revision: str | None = field(default_factory=_str_env("LLM_MODEL_REVISION", None))
    xai_api_key: str | None = field(
        default_factory=_first_env(("XAI_API_KEY", "GROK_API_KEY"), None),
        repr=False,
    )
    xai_model_name: str = field(
        default_factory=_first_env(("XAI_MODEL_NAME", "GROK_MODEL_NAME"), XAI_DEFAULT_MODEL)
    )
    xai_api_url: str = field(
        default_factory=_first_env(("XAI_API_URL", "GROK_API_URL"), XAI_DEFAULT_API_URL)
    )
    xai_timeout_seconds: int = field(default_factory=_int_env("XAI_TIMEOUT_SECONDS", 60))
    ingestion_version: int = field(default_factory=_int_env("INGESTION_VERSION", 1))
    chunker_strategy: str = field(default_factory=_str_env("CHUNKER_STRATEGY", "markdown_v1"))
    chunk_size: int = field(default_factory=_int_env("CHUNK_SIZE", 800))
    chunk_overlap: int = field(default_factory=_int_env("CHUNK_OVERLAP", 100))
    top_k: int = field(default_factory=_int_env("TOP_K", 5))
    relevance_threshold: float = field(default_factory=_float_env("RELEVANCE_THRESHOLD", 0.80))
    # Combine the vector search with BM25 keyword search (see keyword_index.py).
    hybrid_search: bool = field(default_factory=_bool_factory("HYBRID_SEARCH", True))
    # How many chunks each search proposes before the rankings are merged.
    retrieval_candidates: int = field(default_factory=_int_env("RETRIEVAL_CANDIDATES", 30))
    # Weight of the keyword ranking against the vector ranking when they are merged. At
    # equal weight, keyword matches pushed aside the right chunk for English questions
    # about Turkish documents, whose words the Turkish stemmer splits into noise.
    keyword_weight: float = field(default_factory=_float_env("KEYWORD_WEIGHT", 0.5))
    # A cross-encoder that reorders the best candidates (see reranking.py); empty disables it.
    # BAAI/bge-reranker-v2-m3 put every retrievable hard-set answer first but adds about
    # 5 s per question on a CPU, so it is off by default.
    reranker_model_name: str = field(default_factory=_prefix_env("RERANKER_MODEL_NAME", ""))
    reranker_model_revision: str | None = field(default_factory=_str_env("RERANKER_MODEL_REVISION", None))
    rerank_candidates: int = field(default_factory=_int_env("RERANK_CANDIDATES", 20))
    # EasyOCR language codes for PDF pages that have no text layer, comma-separated;
    # empty disables OCR and such documents are rejected as empty.
    ocr_languages: str = field(default_factory=_prefix_env("OCR_LANGUAGES", "tr,en"))
    max_new_tokens: int = field(default_factory=_int_env("MAX_NEW_TOKENS", 256))
    max_upload_mb: int = field(default_factory=_int_env("MAX_UPLOAD_MB", 20))
    # Bytes override the legacy MiB setting when explicitly supplied.
    max_upload_bytes: int | None = field(
        default_factory=_int_env("MAX_UPLOAD_BYTES", None)
    )
    max_http_body_bytes: int = field(default_factory=_int_env("MAX_HTTP_BODY_BYTES", 20 * 1024 * 1024 + 65536))
    max_json_body_bytes: int = field(default_factory=_int_env("MAX_JSON_BODY_BYTES", 32768))
    max_swarm_json_body_bytes: int = field(default_factory=_int_env("MAX_SWARM_JSON_BODY_BYTES", 131072))
    max_multipart_header_bytes: int = field(default_factory=_int_env("MAX_MULTIPART_HEADER_BYTES", 8192))
    max_filename_bytes: int = field(default_factory=_int_env("MAX_FILENAME_BYTES", 512))
    max_question_chars: int = field(default_factory=_int_env("MAX_QUESTION_CHARS", 4000))
    max_pdf_pages: int = field(default_factory=_int_env("MAX_PDF_PAGES", 500))
    max_text_chars: int = field(default_factory=_int_env("MAX_TEXT_CHARS", 2000000))
    max_document_chunks: int = field(default_factory=_int_env("MAX_DOCUMENT_CHUNKS", 2000))
    max_store_chunks: int = field(default_factory=_int_env("MAX_STORE_CHUNKS", 20000))
    max_store_text_bytes: int = field(default_factory=_int_env("MAX_STORE_TEXT_BYTES", 64 * 1024 * 1024))
    max_active_heavy_jobs: int = field(default_factory=_int_env("MAX_ACTIVE_HEAVY_JOBS", 1))
    max_waiting_heavy_jobs: int = field(default_factory=_int_env("MAX_WAITING_HEAVY_JOBS", 2))
    ask_queue_capacity: int = field(default_factory=_int_env("ASK_QUEUE_CAPACITY", 2))
    index_queue_capacity: int = field(default_factory=_int_env("INDEX_QUEUE_CAPACITY", 2))
    model_queue_capacity: int = field(default_factory=_int_env("MODEL_QUEUE_CAPACITY", 2))
    heavy_wait_seconds: int = field(default_factory=_int_env("HEAVY_WAIT_SECONDS", 30))
    preload_models: bool = field(default_factory=lambda: _bool_env("PRELOAD_MODELS"))
    torch_num_threads: int | None = field(default_factory=_int_env("TORCH_NUM_THREADS", None))
    # Relative paths are resolved against the project root.
    index_dir: str = field(default_factory=_str_env("INDEX_DIR", "data/index"))

    def __post_init__(self):
        if self.reranker_model_name and not re.fullmatch(r"[0-9a-f]{40}", self.reranker_model_revision or ""):
            raise ValueError("RERANKER_MODEL_REVISION must be an immutable 40-character commit SHA")
        profile = normalize_profile(self.access_profile)
        object.__setattr__(self, "access_profile", profile)
        hosts = parse_public_hosts(self.public_hosts)
        object.__setattr__(self, "public_hosts", hosts)
        from app.dpapi_vault import unseal_secret

        if self.api_key:
            object.__setattr__(self, "api_key", unseal_secret(self.api_key))
        if self.xai_api_key:
            object.__setattr__(self, "xai_api_key", unseal_secret(self.xai_api_key))

        if profile == "local":
            pass
        elif profile == "hosted":
            if not (self.api_key and self.api_key.strip()):
                raise ValueError("ACCESS_PROFILE=hosted requires EVIDENCERAG_API_KEY")
            if not hosts:
                raise ValueError("ACCESS_PROFILE=hosted requires EVIDENCERAG_PUBLIC_HOST")
            object.__setattr__(self, "api_key", self.api_key.strip())
        else:
            raise ValueError(
                "ACCESS_PROFILE supports 'local' or 'hosted' (alias 'cloud'); "
                "shared/public access is not supported"
            )
        provider = normalize_llm_provider(self.llm_provider)
        object.__setattr__(self, "llm_provider", provider)
        torch_dtype = normalize_llm_torch_dtype(self.llm_torch_dtype)
        object.__setattr__(self, "llm_torch_dtype", torch_dtype)
        validate_xai_model_name(self.xai_model_name)
        validate_xai_api_url(self.xai_api_url)
        if provider == "xai":
            if not (self.xai_api_key and self.xai_api_key.strip()):
                raise ValueError("LLM_PROVIDER=xai requires XAI_API_KEY or GROK_API_KEY")
            object.__setattr__(self, "xai_api_key", self.xai_api_key.strip())
        elif self.xai_api_key:
            object.__setattr__(self, "xai_api_key", self.xai_api_key.strip())
        revision_checks = [("embedding", "embedding_model_name", "embedding_model_revision")]
        if provider == "local":
            revision_checks.append(("llm", "llm_model_name", "llm_model_revision"))
        for kind, name_field, revision_field in revision_checks:
            revision = getattr(self, revision_field)
            if revision is None:
                recorded = model_manifest()[kind]
                if getattr(self, name_field) != recorded["repository"]:
                    raise ValueError(f"{revision_field} is required for a custom model repository")
                revision = recorded["revision"]
                object.__setattr__(self, revision_field, revision)
            if not re.fullmatch(r"[0-9a-f]{40}", revision):
                raise ValueError(f"{revision_field} must be an immutable 40-character commit SHA")
        if self.chunker_strategy not in {"legacy_character","structured_v2","markdown_v1"}:
            raise ValueError("Unsupported CHUNKER_STRATEGY")
        if self.chunker_strategy=="structured_v2" and self.ingestion_version!=2:
            raise ValueError("structured_v2 requires INGESTION_VERSION=2 and a separate rebuilt index")
        if isinstance(self.max_upload_mb, bool) or not isinstance(self.max_upload_mb, int) or self.max_upload_mb <= 0:
            raise ValueError("max_upload_mb must be a positive integer")
        if self.max_upload_bytes is None:
            object.__setattr__(self, "max_upload_bytes", self.max_upload_mb * 1024 * 1024)
        for name in (
            "retrieval_candidates", "rerank_candidates", "chunk_size", "top_k", "max_new_tokens", "max_upload_mb", "max_upload_bytes",
            "max_http_body_bytes", "max_json_body_bytes", "max_swarm_json_body_bytes", "max_multipart_header_bytes",
            "max_filename_bytes", "max_question_chars", "max_pdf_pages", "max_text_chars",
            "max_document_chunks", "max_store_chunks", "max_store_text_bytes",
            "max_active_heavy_jobs", "max_waiting_heavy_jobs", "ask_queue_capacity",
            "index_queue_capacity", "model_queue_capacity", "heavy_wait_seconds",
            "xai_timeout_seconds", "ingestion_version",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.chunk_overlap, bool) or not isinstance(self.chunk_overlap, int) or not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and less than chunk_size")
        if self.torch_num_threads is not None and (not isinstance(self.torch_num_threads, int) or self.torch_num_threads <= 0):
            raise ValueError("TORCH_NUM_THREADS must be a positive integer when set")

    def embedding_signature(self):
        """Settings that determine the stored vectors; changing any of them needs a re-index."""
        return {
            "schema_version": 2,
            "embedding_model_name": self.embedding_model_name,
            "embedding_model_revision": self.embedding_model_revision,
            "tokenizer": {"repository":self.embedding_model_name,"revision":self.embedding_model_revision},
            "embedding_query_prefix": self.embedding_query_prefix,
            "embedding_passage_prefix": self.embedding_passage_prefix,
            "normalization": "l2",
            "embedding_dimension": self.embedding_metadata().get("embedding_dimension"),
            "preprocessing_version": 2,
            "ocr_languages": self.ocr_languages,
            "ingestion": {"strategy":self.chunker_strategy,"version":self.ingestion_version,
                          "chunk_size":self.chunk_size,"chunk_overlap":self.chunk_overlap},
        }

    def embedding_metadata(self):
        record=model_manifest()["embedding"]
        if (self.embedding_model_name,self.embedding_model_revision)==(record["repository"],record["revision"]):
            return {"embedding_dimension":record["embedding_dimension"],"max_sequence_length":record["max_sequence_length"]}
        return {}  # Unknown until an explicitly pinned custom model is actually loaded.

    def model_identity(self):
        if self.llm_provider == "xai":
            llm_identity = {
                "provider": "xai",
                "model": self.xai_model_name,
                "api_url": self.xai_api_url,
            }
        else:
            llm_identity = {
                "repository": self.llm_model_name,
                "revision": self.llm_model_revision,
                "tokenizer_repository": self.llm_model_name,
                "tokenizer_revision": self.llm_model_revision,
                "torch_dtype": self.llm_torch_dtype,
            }
        return {"retrieval": {"hybrid":self.hybrid_search,"candidates":self.retrieval_candidates,
                              "keyword_weight":self.keyword_weight,"reranker":self.reranker_model_name,
                              "reranker_revision":self.reranker_model_revision,"rerank_candidates":self.rerank_candidates},
                "embedding": {"repository":self.embedding_model_name,"revision":self.embedding_model_revision,
                              "tokenizer_repository":self.embedding_model_name,"tokenizer_revision":self.embedding_model_revision},
                "llm": llm_identity,
                "index_signature":self.embedding_signature()}
