import json
import os
import re
import threading
import urllib.error
import urllib.request

from .config import (
    InputLimitError,
    Settings,
    XAI_DEFAULT_API_URL,
    XAI_DEFAULT_MODEL,
    effective_revision,
    normalize_llm_provider,
    normalize_llm_torch_dtype,
    validate_xai_api_url,
    validate_xai_model_name,
)
from .diagnostics import invoke_generator, mark_reason
from .grounding import scrub_url_credentials
from .runtime import SingleFlightCache, configure_torch_threads, get_runtime


class LLMGenerationError(RuntimeError):
    """Raised when the configured language model cannot answer."""


_cache = SingleFlightCache(maxsize=2)
_inference_lock = threading.Lock()
_DEFAULT_LOCAL_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
_MAX_XAI_PROMPT_CHARS = 120_000
_MAX_XAI_RESPONSE_BYTES = 1024 * 1024
_MAX_XAI_TOKENS = 8192
_SECRET_TOKEN_RE = re.compile(
    r"(?:Bearer\s+[A-Za-z0-9._~+/=-]+|xai-[A-Za-z0-9._-]{6,}|sk-[A-Za-z0-9._-]{6,})",
    re.IGNORECASE,
)


def _scrub_secrets(message, secrets=()):
    """Redact API keys, bearer tokens, and URL credentials from error messages."""
    text = scrub_url_credentials(str(message or ""))
    for secret in secrets:
        if secret:
            text = text.replace(secret, "[REDACTED]")
    text = _SECRET_TOKEN_RE.sub("[REDACTED]", text)
    return text


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Block HTTP redirects so Authorization headers cannot be forwarded to third-party/SSRF targets."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise LLMGenerationError("HTTP redirects are disabled for xAI API requests")


def _default_xai_opener():
    return urllib.request.build_opener(_NoRedirectHandler())


def _resolve_torch_dtype(torch_dtype, torch_module):
    normalized = normalize_llm_torch_dtype(torch_dtype)
    if normalized == "auto":
        # Half-precision weights are several times slower on CPUs without native bf16 support.
        return "auto" if torch_module.cuda.is_available() else torch_module.float32
    return getattr(torch_module, normalized)


def get_text_generator(model_name, revision=None, torch_dtype=None):
    revision = effective_revision("llm", model_name, revision)
    resolved_dtype = normalize_llm_torch_dtype(
        torch_dtype if torch_dtype is not None else Settings().llm_torch_dtype
    )
    cache_key = (model_name, revision, resolved_dtype)
    factory = (
        (lambda: _create_text_generator(model_name, revision))
        if torch_dtype is None
        else (lambda: _create_text_generator(model_name, revision, torch_dtype=resolved_dtype))
    )
    return get_runtime().run_sync("load", _cache.get, cache_key, factory)


def _create_text_generator(model_name, revision, torch_dtype=None):
    import torch
    from transformers import pipeline, AutoTokenizer
    configure_torch_threads(torch)

    resolved_dtype = normalize_llm_torch_dtype(
        torch_dtype if torch_dtype is not None else Settings().llm_torch_dtype
    )
    dtype = _resolve_torch_dtype(resolved_dtype, torch)
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    generator = pipeline(
        "text-generation",
        model=model_name,
        tokenizer=tokenizer,
        dtype=dtype,
        revision=revision,
        model_kwargs={"low_cpu_mem_usage": True},
    )
    # Answers should be reproducible, so drop the model's default sampling settings
    # instead of overriding them per call (which makes transformers warn on every request).
    defaults = generator.model.generation_config
    defaults.do_sample = False
    defaults.temperature = None
    defaults.top_p = None
    defaults.top_k = None
    return generator


get_text_generator.cache_clear = _cache.clear


def validate_prompt_budget(messages, tokenizer, context_window, max_new_tokens):
    tokens = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    length = len(tokens)
    if length + max_new_tokens > context_window:
        raise InputLimitError(f"Prompt uses {length} tokens plus {max_new_tokens} answer tokens; context limit is {context_window}")
    return length


def _context_window(generator):
    config = getattr(getattr(generator, "model", None), "config", None)
    candidates = [getattr(config, key, None) for key in ["max_position_embeddings", "n_positions"]]
    tokenizer_limit = getattr(getattr(generator, "tokenizer", None), "model_max_length", None)
    if isinstance(tokenizer_limit, int) and 0 < tokenizer_limit < 2**31:
        candidates.append(tokenizer_limit)
    candidates = [v for v in candidates if isinstance(v, int) and v > 0]
    if not candidates:
        raise LLMGenerationError("Model/tokenizer context limit is unavailable")
    return min(candidates)


def _validate_xai_messages_budget(messages, max_prompt_chars=_MAX_XAI_PROMPT_CHARS):
    if not isinstance(messages, list):
        raise LLMGenerationError("Messages must be a list")
    total_chars = 0
    for message in messages:
        if not isinstance(message, dict):
            raise LLMGenerationError("Each message must be a dictionary")
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"} or not isinstance(content, str):
            raise LLMGenerationError("Invalid chat message structure for xAI API")
        total_chars += len(content)
    if total_chars > max_prompt_chars:
        raise InputLimitError(
            f"Prompt uses {total_chars} characters; xAI prompt limit is {max_prompt_chars}"
        )
    return total_chars


def generate_with_xai(
    messages,
    *,
    model_name=XAI_DEFAULT_MODEL,
    api_key=None,
    api_url=XAI_DEFAULT_API_URL,
    max_new_tokens=256,
    timeout_seconds=60,
    max_prompt_chars=_MAX_XAI_PROMPT_CHARS,
    opener=None,
):
    """Invoke the xAI Grok chat completions API securely without leaking credentials."""
    resolved_key = (api_key or os.getenv("XAI_API_KEY") or os.getenv("GROK_API_KEY") or "").strip()
    secrets = (resolved_key,) if resolved_key else ()
    try:
        validated_model = validate_xai_model_name(model_name)
        validated_url = validate_xai_api_url(api_url)
        if not resolved_key:
            raise LLMGenerationError("xAI API key is required (set XAI_API_KEY or GROK_API_KEY)")
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or not 1 <= max_new_tokens <= _MAX_XAI_TOKENS:
            raise LLMGenerationError(f"max_new_tokens must be between 1 and {_MAX_XAI_TOKENS}")
        if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)) or timeout_seconds <= 0:
            raise LLMGenerationError("timeout_seconds must be positive")
        _validate_xai_messages_budget(messages, max_prompt_chars=max_prompt_chars)

        payload = json.dumps(
            {
                "model": validated_model,
                "messages": messages,
                "max_tokens": max_new_tokens,
                "temperature": 0,
                "stream": False,
            },
            ensure_ascii=False,
        ).encode("utf-8")

        request = urllib.request.Request(
            validated_url,
            data=payload,
            method="POST",
            headers={
                "Authorization": f"Bearer {resolved_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "EvidenceRAG/1.0",
            },
        )
        http_opener = opener if opener is not None else _default_xai_opener()

        def _perform_request():
            with http_opener.open(request, timeout=timeout_seconds) as response:
                raw = response.read(_MAX_XAI_RESPONSE_BYTES + 1)
                if len(raw) > _MAX_XAI_RESPONSE_BYTES:
                    raise LLMGenerationError("xAI API response exceeded size limit")
                return raw

        raw_body = invoke_generator(_perform_request)
        try:
            data = json.loads(raw_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LLMGenerationError("xAI API returned malformed JSON") from exc

        choices = data.get("choices") if isinstance(data, dict) else None
        if not isinstance(choices, list) or not choices:
            raise LLMGenerationError("xAI API response contained no choices")
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise LLMGenerationError("xAI API response contained no text content")
        return content.strip()
    except InputLimitError:
        mark_reason("input_limit")
        raise
    except urllib.error.HTTPError as exc:
        mark_reason("model_error")
        status = getattr(exc, "code", "unknown")
        raise LLMGenerationError(f"xAI API request failed with HTTP {status}") from None
    except urllib.error.URLError as exc:
        mark_reason("model_error")
        safe_reason = _scrub_secrets(getattr(exc, "reason", exc), secrets=secrets)
        raise LLMGenerationError(f"xAI API network error: {safe_reason}") from None
    except LLMGenerationError as exc:
        mark_reason("model_error")
        raise LLMGenerationError(_scrub_secrets(exc, secrets=secrets)) from None
    except Exception as exc:
        mark_reason("model_error")
        raise LLMGenerationError(
            f"xAI generation failed: {_scrub_secrets(exc, secrets=secrets)}"
        ) from None


def generate_answer(
    messages,
    model_name=None,
    max_new_tokens=256,
    text_generator=None,
    revision=None,
    *,
    provider="local",
    torch_dtype=None,
    api_key=None,
    api_url=XAI_DEFAULT_API_URL,
    timeout_seconds=60,
    opener=None,
):
    resolved_provider = normalize_llm_provider(provider)
    resolved_torch_dtype = normalize_llm_torch_dtype(torch_dtype) if torch_dtype is not None else None
    if text_generator is None and resolved_provider == "xai":
        resolved_model = model_name or XAI_DEFAULT_MODEL
        return get_runtime().run_sync(
            "generation",
            generate_with_xai,
            messages,
            model_name=resolved_model,
            api_key=api_key,
            api_url=api_url,
            max_new_tokens=max_new_tokens,
            timeout_seconds=timeout_seconds,
            opener=opener,
        )
    resolved_model = model_name or _DEFAULT_LOCAL_MODEL
    return get_runtime().run_sync(
        "generation",
        _generate_answer,
        messages,
        resolved_model,
        max_new_tokens,
        text_generator,
        revision,
        resolved_torch_dtype,
    )


def _generate_answer(messages, model_name, max_new_tokens, text_generator, revision, torch_dtype=None):
    try:
        generator = (
            text_generator
            if text_generator is not None
            else get_text_generator(model_name, revision=revision, torch_dtype=torch_dtype)
        )
        tokenizer = getattr(generator, "tokenizer", None)
        if tokenizer is None:
            raise LLMGenerationError("Tokenizer is required for the complete prompt budget")
        with _inference_lock:
            validate_prompt_budget(messages, tokenizer, _context_window(generator), max_new_tokens)
            if text_generator is None:
                from transformers import GenerationConfig
                options={"generation_config":GenerationConfig(max_new_tokens=max_new_tokens,do_sample=False)}
            else:
                # Explicit injected callable protocol uses primitives, no Transformers dependency.
                options={"max_new_tokens":max_new_tokens,"do_sample":False}
            output = invoke_generator(generator,
                messages,
                **options,
                return_full_text=False,
                clean_up_tokenization_spaces=False,
            )
        return output[0]["generated_text"].strip()
    except InputLimitError:
        mark_reason("input_limit")
        raise
    except Exception as exc:
        mark_reason("model_error")
        raise LLMGenerationError(f"LLM generation failed: {_scrub_secrets(exc)}") from exc
