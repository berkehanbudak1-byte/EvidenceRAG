"""Zero-extra-dependency parallel multi-agent Swarm Orchestration for EvidenceRAG.

Built purely on Python's standard library ``concurrent.futures.ThreadPoolExecutor``
with strict security invariants:
- Disjoint slice ownership (no two workers may own overlapping sources/chunks)
- Read-only worker tool allowlist (write/delete/egress tools are strictly blocked)
- Bounded worker concurrency and per-slice execution timeout
- ``<untrusted_evidence>`` and ``<untrusted_data>`` isolation via ``app.rag.grounding``
- Indirect prompt-injection detection & ``one_door_out`` human-approval escalation
- Traceability verification via ``verify_number_traceability`` + ``verify_answer``
- Provider routing supporting ``local`` and ``xai`` (alias ``grok``)
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
import os
import re
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

from .config import (
    XAI_DEFAULT_API_URL,
    XAI_DEFAULT_MODEL,
    normalize_llm_provider,
    validate_xai_api_url,
    validate_xai_model_name,
)
from .generation import _scrub_secrets, generate_answer
from .grounding import (
    calculate_evidence_confidence,
    detect_prompt_injection,
    inspect_evidence_for_injection,
    verify_answer,
    verify_number_traceability,
    wrap_untrusted_evidence,
)
from .prompt import build_messages

READ_ONLY_WORKER_TOOLS = frozenset(
    {
        "retrieve",
        "read_chunk",
        "verify_citations",
        "inspect_injection",
        "extract_claims",
        "trace_numbers",
        "summarize_slice",
    }
)

FORBIDDEN_WRITE_TOOLS = frozenset(
    {
        "write_file",
        "delete_document",
        "index_document",
        "modify_store",
        "shell_exec",
        "http_post",
        "network_egress",
        "send_webhook",
    }
)

ONE_DOOR_OUT_ACTIONS = frozenset(
    {
        "delete_document",
        "modify_store",
        "external_egress",
        "publish_report",
        "webhook_emit",
    }
)

MAX_SWARM_WORKERS = 8
DEFAULT_SWARM_WORKERS = 4
MAX_SWARM_SLICES = 16
DEFAULT_SLICE_TIMEOUT_SECONDS = 15.0
MAX_SLICE_TIMEOUT_SECONDS = 60.0
MAX_QUESTION_CHARS = 4000
MAX_SLICE_EVIDENCE_ITEMS = 64

_SAFE_ID_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._:-]{0,63}$")


class SwarmPolicyError(ValueError):
    """Raised when a swarm work order or slice violates security invariants."""


@dataclass(frozen=True)
class SwarmSlice:
    """An isolated, read-only unit of work owned by a single swarm worker."""

    slice_id: str
    role: str = "evidence_verifier"
    question: str = ""
    evidence_items: tuple[dict[str, Any], ...] | list[dict[str, Any]] = field(default_factory=list)
    owned_sources: tuple[str, ...] | list[str] = field(default_factory=tuple)
    requested_tools: tuple[str, ...] | list[str] = (
        "read_chunk",
        "verify_citations",
        "inspect_injection",
        "trace_numbers",
    )
    requested_actions: tuple[str, ...] | list[str] = field(default_factory=tuple)
    timeout_seconds: float | None = None


@dataclass(frozen=True)
class SwarmWorkOrder:
    """Validated work order governing a parallel read-only EvidenceRAG swarm run."""

    order_id: str
    question: str
    slices: tuple[SwarmSlice, ...] | list[SwarmSlice]
    provider: str = "local"
    model_name: str | None = None
    api_url: str = XAI_DEFAULT_API_URL
    max_workers: int = DEFAULT_SWARM_WORKERS
    slice_timeout_seconds: float = DEFAULT_SLICE_TIMEOUT_SECONDS
    max_new_tokens: int = 256
    requested_actions: tuple[str, ...] | list[str] = field(default_factory=tuple)
    require_traceability: bool = True


def _normalize_id(value: Any, label: str) -> str:
    text = str(value or "").strip()
    if not text or not _SAFE_ID_RE.fullmatch(text):
        raise SwarmPolicyError(
            f"Invalid {label}: must match ^[a-zA-Z0-9][a-zA-Z0-9._:-]{{0,63}}$"
        )
    return text


def _normalize_timeout(value: Any, label: str = "slice_timeout_seconds") -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SwarmPolicyError(f"{label} must be a positive number")
    timeout = float(value)
    if timeout <= 0.0 or timeout > MAX_SLICE_TIMEOUT_SECONDS:
        raise SwarmPolicyError(
            f"{label} must be within (0, {MAX_SLICE_TIMEOUT_SECONDS}] seconds"
        )
    return timeout


def enforce_worker_read_only_tools(
    requested_tools: Iterable[str],
    *,
    allowed_tools: frozenset[str] = READ_ONLY_WORKER_TOOLS,
) -> tuple[str, ...]:
    """Enforce that swarm workers only declare allowlisted read-only tools."""
    if requested_tools is None or isinstance(requested_tools, (str, bytes)):
        raise SwarmPolicyError("requested_tools must be a sequence of tool names")

    validated: list[str] = []
    for raw_tool in requested_tools:
        if not isinstance(raw_tool, str) or not raw_tool.strip():
            raise SwarmPolicyError("Tool name must be a non-empty string")
        tool = raw_tool.strip().lower()
        if tool in FORBIDDEN_WRITE_TOOLS:
            raise SwarmPolicyError(
                f"Worker tool '{tool}' is forbidden: workers are strictly read-only"
            )
        if tool not in allowed_tools:
            raise SwarmPolicyError(
                f"Worker tool '{tool}' is not in the read-only allowlist"
            )
        if tool not in validated:
            validated.append(tool)

    if not validated:
        raise SwarmPolicyError("At least one read-only worker tool must be declared")
    return tuple(validated)


def _derive_slice_ownership_keys(slice_obj: SwarmSlice) -> list[str]:
    """Derive canonical ownership keys for a slice from explicit owned_sources or evidence."""
    explicit = [str(s).strip().lower() for s in (slice_obj.owned_sources or ()) if str(s).strip()]
    if explicit:
        # Preserve order while deduplicating within a single slice.
        return list(dict.fromkeys(explicit))

    derived: list[str] = []
    for idx, item in enumerate(slice_obj.evidence_items or (), start=1):
        if isinstance(item, Mapping):
            key = (
                item.get("chunk_id")
                or item.get("id")
                or item.get("source")
                or item.get("filename")
                or f"{slice_obj.slice_id}:item:{idx}"
            )
        else:
            key = f"{slice_obj.slice_id}:item:{idx}"
        norm_key = str(key).strip().lower()
        if norm_key and norm_key not in derived:
            derived.append(norm_key)

    if not derived:
        derived.append(f"slice:{slice_obj.slice_id.strip().lower()}")
    return derived


def validate_disjoint_slices(slices: Sequence[SwarmSlice]) -> dict[str, list[str]]:
    """Validate bounded slice count, unique slice IDs, and strictly disjoint slice ownership."""
    if not isinstance(slices, (list, tuple)) or not slices:
        raise SwarmPolicyError("Swarm work order must contain at least one slice")
    if len(slices) > MAX_SWARM_SLICES:
        raise SwarmPolicyError(f"Slice count exceeds maximum of {MAX_SWARM_SLICES}")

    seen_slice_ids: set[str] = set()
    owner_by_key: dict[str, str] = {}
    ownership_map: dict[str, list[str]] = {}

    for slice_obj in slices:
        if not isinstance(slice_obj, SwarmSlice):
            raise SwarmPolicyError("Every slice must be a SwarmSlice instance")
        slice_id = _normalize_id(slice_obj.slice_id, "slice_id")
        if slice_id in seen_slice_ids:
            raise SwarmPolicyError(f"Duplicate slice_id detected: '{slice_id}'")
        seen_slice_ids.add(slice_id)

        evidence_items = list(slice_obj.evidence_items or ())
        if len(evidence_items) > MAX_SLICE_EVIDENCE_ITEMS:
            raise SwarmPolicyError(
                f"Slice '{slice_id}' exceeds maximum of {MAX_SLICE_EVIDENCE_ITEMS} evidence items"
            )

        keys = _derive_slice_ownership_keys(slice_obj)
        for key in keys:
            previous_owner = owner_by_key.get(key)
            if previous_owner is not None and previous_owner != slice_id:
                raise SwarmPolicyError(
                    f"Disjoint slice violation: resource '{key}' is claimed by both "
                    f"'{previous_owner}' and '{slice_id}'"
                )
            owner_by_key[key] = slice_id
        ownership_map[slice_id] = keys

    return ownership_map


def verify_traceability(
    answer: str,
    evidence_items: Sequence[Mapping[str, Any] | str],
    *,
    question: str = "",
) -> dict[str, Any]:
    """Verify numeric traceability and citation grounding against input evidence."""
    normalized_evidence: list[dict[str, Any]] = []
    for idx, item in enumerate(evidence_items or (), start=1):
        if isinstance(item, Mapping):
            entry = dict(item)
            entry.setdefault("text", "")
            entry.setdefault("source", entry.get("filename") or f"evidence_{idx}")
        else:
            entry = {"text": str(item or ""), "source": f"evidence_{idx}"}
        normalized_evidence.append(entry)

    num_report = verify_number_traceability(answer, normalized_evidence)
    ans_report = verify_answer(str(answer or ""), normalized_evidence, question=question)
    confidence = calculate_evidence_confidence(normalized_evidence, ans_report)
    passed = bool(num_report["done_when_passed"]) and not ans_report["invalid_citations"]
    return {
        "done_when_passed": passed,
        "traceability_ratio": num_report["traceability_ratio"],
        "traced": num_report["traced"],
        "untraced": num_report["untraced"],
        "citations": ans_report["citations"],
        "invalid_citations": ans_report["invalid_citations"],
        "unsupported_numbers": ans_report["unsupported_numbers"],
        "question_only_numbers": ans_report["question_only_numbers"],
        "confidence": confidence,
    }


def isolate_slice_evidence(
    slice_id: str,
    evidence_items: Sequence[Mapping[str, Any] | str],
) -> dict[str, Any]:
    """Wrap slice evidence in <untrusted_evidence>/<untrusted_data> envelopes with injection flags."""
    safe_slice_id = _normalize_id(slice_id, "slice_id")
    wrapped_result = wrap_untrusted_evidence(evidence_items)
    inspection = inspect_evidence_for_injection(evidence_items)

    inner_blocks = "\n\n".join(item["wrapped"] for item in wrapped_result["items"])
    envelope = (
        f'<untrusted_evidence slice_id="{safe_slice_id}" '
        f'count="{len(wrapped_result["items"])}" '
        f'injection_flagged_count="{wrapped_result["flagged_count"]}">\n'
        f"{inner_blocks}\n"
        f"</untrusted_evidence>"
    )
    return {
        "slice_id": safe_slice_id,
        "envelope": envelope,
        "items": wrapped_result["items"],
        "flagged_count": wrapped_result["flagged_count"],
        "safe": wrapped_result["safe"] and not inspection["detected"],
        "inspection": inspection,
    }


def evaluate_one_door_out(
    slice_results: Sequence[Mapping[str, Any]],
    *,
    requested_actions: Iterable[str] = (),
    question_injection: Mapping[str, Any] | None = None,
    require_traceability: bool = True,
) -> dict[str, Any]:
    """Evaluate the one_door_out approval gate across all swarm outputs and requested actions.

    Workers cannot execute privileged actions or egress autonomously; any privileged
    action, prompt-injection finding, slice failure/timeout, or failed traceability
    escalates cleanly to human approval.
    """
    reasons: list[str] = []
    gated_actions: list[str] = []

    for raw_action in requested_actions or ():
        action = str(raw_action or "").strip().lower()
        if not action:
            continue
        if action not in gated_actions:
            gated_actions.append(action)
        if action in ONE_DOOR_OUT_ACTIONS or action not in READ_ONLY_WORKER_TOOLS:
            reasons.append(f"privileged_action:{action}")

    if question_injection and question_injection.get("detected"):
        reasons.append("question_prompt_injection_detected")

    for res in slice_results:
        sid = res.get("slice_id", "unknown")
        for raw_action in res.get("requested_actions") or ():
            action = str(raw_action or "").strip().lower()
            if action and action not in gated_actions:
                gated_actions.append(action)
            if action:
                reasons.append(f"slice:{sid}:privileged_action:{action}")

        if res.get("injection_detected"):
            reasons.append(f"slice:{sid}:prompt_injection_detected")

        status = res.get("status")
        if status != "ok":
            reasons.append(f"slice:{sid}:status:{status}")

        if require_traceability and res.get("traceability") and not res["traceability"].get("done_when_passed"):
            reasons.append(f"slice:{sid}:untraced_output")

    deduped_reasons = list(dict.fromkeys(reasons))
    approval_required = bool(deduped_reasons)
    return {
        "gate": "one_door_out",
        "approval_required": approval_required,
        "status": "ESCALATED_FOR_APPROVAL" if approval_required else "READ_ONLY_COMPLETE",
        "autonomous_egress_blocked": True,
        "gated_actions": gated_actions,
        "reasons": deduped_reasons,
    }


def _deterministic_local_summary(slice_obj: SwarmSlice, isolated: Mapping[str, Any]) -> str:
    """Read-only offline evidence synthesis when no live LLM generator is invoked."""
    items = isolated.get("items") or []
    if not items:
        return "NO_EVIDENCE"
    parts: list[str] = []
    for idx, item in enumerate(items, start=1):
        clean_text = str(item.get("clean_text", "") or "").strip()
        if not clean_text:
            continue
        snippet = clean_text[:220].rstrip()
        parts.append(f"{snippet} [Kanıt {idx}]")
    return " ".join(parts) if parts else "NO_EVIDENCE"


def get_swarm_policy() -> dict[str, Any]:
    """Return static, non-sensitive swarm capabilities and security policy metadata."""
    return {
        "engine": "stdlib.concurrent.futures.ThreadPoolExecutor",
        "zero_extra_dependencies": True,
        "max_workers": MAX_SWARM_WORKERS,
        "default_workers": DEFAULT_SWARM_WORKERS,
        "max_slices": MAX_SWARM_SLICES,
        "max_slice_evidence_items": MAX_SLICE_EVIDENCE_ITEMS,
        "max_slice_timeout_seconds": MAX_SLICE_TIMEOUT_SECONDS,
        "supported_providers": ["local", "xai", "grok"],
        "read_only_worker_tools": sorted(READ_ONLY_WORKER_TOOLS),
        "forbidden_write_tools": sorted(FORBIDDEN_WRITE_TOOLS),
        "one_door_out_actions": sorted(ONE_DOOR_OUT_ACTIONS),
        "invariants": [
            "disjoint_slice_ownership",
            "read_only_worker_tool_allowlist",
            "bounded_concurrency",
            "per_slice_timeout",
            "untrusted_evidence_isolation",
            "prompt_injection_detection",
            "one_door_out_approval_escalation",
            "number_and_citation_traceability",
        ],
    }


def run_swarm_orchestration(
    work_order: SwarmWorkOrder,
    *,
    worker_fn: Callable[[SwarmSlice, dict[str, Any]], str] | None = None,
    text_generator: Any = None,
    api_key: str | None = None,
    opener: Any = None,
) -> dict[str, Any]:
    """Execute a validated EvidenceRAG SwarmWorkOrder across isolated read-only worker threads."""
    if not isinstance(work_order, SwarmWorkOrder):
        raise SwarmPolicyError("work_order must be a SwarmWorkOrder instance")

    order_id = _normalize_id(work_order.order_id, "order_id")
    question = str(work_order.question or "").strip()
    if not question:
        raise SwarmPolicyError("Work order question must not be empty")
    if len(question) > MAX_QUESTION_CHARS:
        raise SwarmPolicyError(f"Question exceeds maximum length of {MAX_QUESTION_CHARS} characters")

    if isinstance(work_order.max_workers, bool) or not isinstance(work_order.max_workers, int):
        raise SwarmPolicyError("max_workers must be an integer")
    if not 1 <= work_order.max_workers <= MAX_SWARM_WORKERS:
        raise SwarmPolicyError(f"max_workers must be between 1 and {MAX_SWARM_WORKERS}")

    default_timeout = _normalize_timeout(work_order.slice_timeout_seconds, "slice_timeout_seconds")

    try:
        resolved_provider = normalize_llm_provider(work_order.provider)
    except ValueError as exc:
        raise SwarmPolicyError(str(exc)) from exc

    resolved_model = work_order.model_name
    validated_api_url = work_order.api_url
    if resolved_provider == "xai":
        resolved_model = resolved_model or XAI_DEFAULT_MODEL
        try:
            resolved_model = validate_xai_model_name(resolved_model)
            validated_api_url = validate_xai_api_url(work_order.api_url)
        except ValueError as exc:
            raise SwarmPolicyError(str(exc)) from exc
    else:
        resolved_model = resolved_model or "local-deterministic"

    ownership_map = validate_disjoint_slices(work_order.slices)

    validated_tools_by_slice: dict[str, tuple[str, ...]] = {}
    timeouts_by_slice: dict[str, float] = {}
    for slice_obj in work_order.slices:
        sid = slice_obj.slice_id.strip()
        validated_tools_by_slice[sid] = enforce_worker_read_only_tools(slice_obj.requested_tools)
        timeouts_by_slice[sid] = (
            _normalize_timeout(slice_obj.timeout_seconds, f"slice:{sid}.timeout_seconds")
            if slice_obj.timeout_seconds is not None
            else default_timeout
        )

    question_injection = detect_prompt_injection(question)
    effective_workers = min(work_order.max_workers, len(work_order.slices))
    resolved_xai_key = (api_key or os.getenv("XAI_API_KEY") or os.getenv("GROK_API_KEY") or "").strip()
    secrets = (resolved_xai_key,) if resolved_xai_key else ()

    def _execute_slice(slice_obj: SwarmSlice) -> dict[str, Any]:
        started = time.monotonic()
        sid = slice_obj.slice_id.strip()
        slice_question = str(slice_obj.question or question).strip()
        evidence_items = [
            dict(item) if isinstance(item, Mapping) else {"text": str(item or "")}
            for item in (slice_obj.evidence_items or ())
        ]
        isolated = isolate_slice_evidence(sid, evidence_items)
        slice_q_injection = detect_prompt_injection(slice_question)
        injection_detected = (not isolated["safe"]) or bool(slice_q_injection["detected"])

        worker_context = {
            "order_id": order_id,
            "slice_id": sid,
            "role": slice_obj.role,
            "question": slice_question,
            "provider": resolved_provider,
            "model_name": resolved_model,
            "allowed_tools": validated_tools_by_slice[sid],
            "owned_sources": ownership_map[sid],
            "untrusted_envelope": isolated["envelope"],
            "wrapped_items": isolated["items"],
            "injection_detected": injection_detected,
        }

        if worker_fn is not None:
            raw_output = str(worker_fn(slice_obj, worker_context) or "").strip()
        elif text_generator is not None or (
            resolved_provider == "xai" and (resolved_xai_key or opener is not None)
        ):
            messages = build_messages(slice_question, evidence_items)
            raw_output = generate_answer(
                messages,
                model_name=resolved_model,
                max_new_tokens=work_order.max_new_tokens,
                text_generator=text_generator,
                provider=resolved_provider,
                api_key=resolved_xai_key or None,
                api_url=validated_api_url,
                timeout_seconds=timeouts_by_slice[sid],
                opener=opener,
            )
        else:
            raw_output = _deterministic_local_summary(slice_obj, isolated)

        traceability = verify_traceability(
            raw_output,
            evidence_items,
            question=slice_question,
        )
        return {
            "slice_id": sid,
            "role": slice_obj.role,
            "status": "ok",
            "provider": resolved_provider,
            "model_name": resolved_model,
            "allowed_tools": list(validated_tools_by_slice[sid]),
            "owned_sources": ownership_map[sid],
            "requested_actions": [str(a).strip().lower() for a in (slice_obj.requested_actions or ()) if str(a).strip()],
            "untrusted_envelope": isolated["envelope"],
            "injection_detected": injection_detected,
            "injection_flagged_count": isolated["flagged_count"],
            "injection_warnings": isolated["inspection"]["warnings"],
            "output": raw_output,
            "traceability": traceability,
            "duration_ms": round((time.monotonic() - started) * 1000.0, 2),
        }

    slice_results: list[dict[str, Any]] = []
    executor = ThreadPoolExecutor(max_workers=effective_workers, thread_name_prefix="evidencerag-swarm")
    try:
        submitted = [
            (slice_obj, executor.submit(_execute_slice, slice_obj))
            for slice_obj in work_order.slices
        ]
        for slice_obj, future in submitted:
            sid = slice_obj.slice_id.strip()
            timeout = timeouts_by_slice[sid]
            try:
                slice_results.append(future.result(timeout=timeout))
            except FutureTimeoutError:
                future.cancel()
                slice_results.append(
                    {
                        "slice_id": sid,
                        "role": slice_obj.role,
                        "status": "timeout",
                        "provider": resolved_provider,
                        "model_name": resolved_model,
                        "allowed_tools": list(validated_tools_by_slice[sid]),
                        "owned_sources": ownership_map[sid],
                        "requested_actions": [
                            str(a).strip().lower()
                            for a in (slice_obj.requested_actions or ())
                            if str(a).strip()
                        ],
                        "injection_detected": False,
                        "injection_flagged_count": 0,
                        "injection_warnings": [],
                        "output": "",
                        "error": f"Slice '{sid}' exceeded timeout of {timeout}s",
                        "traceability": None,
                        "duration_ms": round(timeout * 1000.0, 2),
                    }
                )
            except Exception as exc:
                safe_msg = _scrub_secrets(exc, secrets=secrets)
                slice_results.append(
                    {
                        "slice_id": sid,
                        "role": slice_obj.role,
                        "status": "error",
                        "provider": resolved_provider,
                        "model_name": resolved_model,
                        "allowed_tools": list(validated_tools_by_slice[sid]),
                        "owned_sources": ownership_map[sid],
                        "requested_actions": [
                            str(a).strip().lower()
                            for a in (slice_obj.requested_actions or ())
                            if str(a).strip()
                        ],
                        "injection_detected": False,
                        "injection_flagged_count": 0,
                        "injection_warnings": [],
                        "output": "",
                        "error": safe_msg,
                        "traceability": None,
                        "duration_ms": 0.0,
                    }
                )
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    all_evidence: list[dict[str, Any]] = []
    merged_parts: list[str] = []
    raw_outputs: list[str] = []
    for slice_obj, res in zip(work_order.slices, slice_results):
        for item in slice_obj.evidence_items or ():
            all_evidence.append(dict(item) if isinstance(item, Mapping) else {"text": str(item or "")})
        if res["status"] == "ok" and res.get("output"):
            merged_parts.append(f"[{res['slice_id']}] {res['output']}")
            raw_outputs.append(res["output"])

    merged_summary = "\n".join(merged_parts)
    corpus_traceability = verify_traceability("\n".join(raw_outputs), all_evidence, question=question)
    flagged_slices = [r["slice_id"] for r in slice_results if r.get("injection_detected")]
    injection_summary = {
        "detected": bool(flagged_slices) or bool(question_injection["detected"]),
        "question_flagged": bool(question_injection["detected"]),
        "question_rules": list(question_injection["rules"]),
        "flagged_slices": flagged_slices,
        "flagged_chunk_count": sum(int(r.get("injection_flagged_count", 0)) for r in slice_results),
    }

    one_door_out = evaluate_one_door_out(
        slice_results,
        requested_actions=work_order.requested_actions,
        question_injection=question_injection,
        require_traceability=work_order.require_traceability,
    )

    return {
        "order_id": order_id,
        "question": question,
        "provider": resolved_provider,
        "model_name": resolved_model,
        "max_workers": effective_workers,
        "slice_timeout_seconds": default_timeout,
        "disjoint_ownership": ownership_map,
        "slices": slice_results,
        "merged_summary": merged_summary,
        "injection_summary": injection_summary,
        "traceability": corpus_traceability,
        "one_door_out": one_door_out,
        "security_invariants": get_swarm_policy()["invariants"],
    }
