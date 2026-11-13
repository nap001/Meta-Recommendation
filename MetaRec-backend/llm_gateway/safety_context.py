"""Safety context producers and automatic call-site policy metadata inference.

Implements ``summary_safety_context.v1`` and ``confirmation_context.v1`` inside
``llm_gateway`` so upstream application modules (``conversation_context.py``,
``llm_service.py``, ``service.py``, ``agent/``) require zero modifications.
"""

from __future__ import annotations

import inspect
import time
from typing import Any, Dict, List, Optional

from llm_gateway.constants import (
    DEFAULT_CONTEXT_VERSION,
    POLICY_TAG_AGENT_PLAN,
    POLICY_TAG_AGENT_SUMMARY,
    POLICY_TAG_CONFIRMATION_PAYLOAD,
    POLICY_TAG_CONVERSATION_SUMMARY,
    POLICY_TAG_GATHER_ACTION,
    POLICY_TAG_INTENT_ANALYZE,
    POLICY_TAG_ITINERARY_CONSTRAINTS,
    POLICY_TAG_ITINERARY_CONTAINMENT,
    POLICY_TAG_ITINERARY_DURATIONS,
    POLICY_TAG_ITINERARY_REPAIR,
    POLICY_TAG_ITINERARY_ROLES,
    PRODUCER_ID_CONFIRMATION_CONTEXT,
    PRODUCER_ID_SUMMARY_SAFETY,
)

SUMMARY_SAFETY_PRODUCER_ID: str = PRODUCER_ID_SUMMARY_SAFETY
SUMMARY_SAFETY_CONTEXT_VERSION: str = DEFAULT_CONTEXT_VERSION

CONFIRMATION_CONTEXT_PRODUCER_ID: str = PRODUCER_ID_CONFIRMATION_CONTEXT
CONFIRMATION_CONTEXT_VERSION: str = DEFAULT_CONTEXT_VERSION

_ALLOWED_ROUTE_MODES = frozenset({"single_domain", "multi_domain", "itinerary", "domain_error"})

# Maps the 11 MetaRec call-site function names to their canonical WP5 policy_tag
CALL_SITE_TAG_MAP: Dict[str, str] = {
    "summarize_conversation": POLICY_TAG_CONVERSATION_SUMMARY,
    "generate_confirmation_payload": POLICY_TAG_CONFIRMATION_PAYLOAD,
    "propose_gather_action": POLICY_TAG_GATHER_ACTION,
    "analyze_user_message": POLICY_TAG_INTENT_ANALYZE,
    "extract_itinerary_constraints": POLICY_TAG_ITINERARY_CONSTRAINTS,
    "enrich_itinerary_durations": POLICY_TAG_ITINERARY_DURATIONS,
    "classify_itinerary_candidate_roles": POLICY_TAG_ITINERARY_ROLES,
    "classify_itinerary_containment": POLICY_TAG_ITINERARY_CONTAINMENT,
    "propose_itinerary_repair": POLICY_TAG_ITINERARY_REPAIR,
    "run_demo": POLICY_TAG_AGENT_PLAN,
    "summarize_recommendations": POLICY_TAG_AGENT_SUMMARY,
}


def _message_has_itinerary_markers(metadata: Dict[str, Any]) -> bool:
    if metadata.get("domain") == "itinerary" or metadata.get("graph") == "itinerary_graph":
        return True
    if metadata.get("itinerary"):
        return True
    conf_req = metadata.get("confirmation_request")
    if isinstance(conf_req, dict):
        prefs = conf_req.get("preferences")
        if isinstance(prefs, dict) and prefs.get("domain") == "itinerary":
            return True
    rec_data = metadata.get("recommendation_data")
    if isinstance(rec_data, dict):
        rec_meta = rec_data.get("metadata")
        if isinstance(rec_meta, dict) and (
            rec_meta.get("domain") == "itinerary" or rec_meta.get("graph") == "itinerary_graph"
        ):
            return True
    return False


def build_summary_safety_context(
    messages: List[Any],
    *,
    conversation_metadata: Optional[Dict[str, Any]] = None,
    now_utc: Optional[float] = None,
) -> Dict[str, Any]:
    """Derive ``summary_safety_context.v1`` from persisted message metadata before flattening.

    Per WP5 specification:
    - Inspects ``metadata["domain"] == "itinerary"``, ``metadata["graph"] == "itinerary_graph"``,
      and ``metadata["itinerary"]`` on messages being folded into the summary (plus prior
      summary safety metadata if present).
    - Any relevant message that cannot be classified yields ``safety_class="unknown"`` and
      fails closed to the baseline route.
    """
    produced_at = time.time() if now_utc is None else float(now_utc)
    if not isinstance(messages, list) or not messages:
        return {
            "contains_itinerary_state": True,
            "request_mode": "itinerary",
            "safety_class": "unknown",
            "producer_id": SUMMARY_SAFETY_PRODUCER_ID,
            "context_version": SUMMARY_SAFETY_CONTEXT_VERSION,
            "produced_at_utc": produced_at,
        }

    if isinstance(conversation_metadata, dict):
        ctx_summary = conversation_metadata.get("context_summary")
        if isinstance(ctx_summary, dict) and ctx_summary.get("contains_itinerary_state") is True:
            return {
                "contains_itinerary_state": True,
                "request_mode": "itinerary",
                "safety_class": "itinerary",
                "producer_id": SUMMARY_SAFETY_PRODUCER_ID,
                "context_version": SUMMARY_SAFETY_CONTEXT_VERSION,
                "produced_at_utc": produced_at,
            }

    has_itinerary = False
    has_multi_domain = False
    has_unknown = False

    for msg in messages:
        if not isinstance(msg, dict):
            has_unknown = True
            break
        role = msg.get("role")
        if role not in {"user", "assistant"}:
            has_unknown = True
            break
        raw_meta = msg.get("metadata")
        if raw_meta is not None and not isinstance(raw_meta, dict):
            has_unknown = True
            break
        metadata = raw_meta if isinstance(raw_meta, dict) else {}
        if metadata.get("type") == "processing":
            continue
        if _message_has_itinerary_markers(metadata):
            has_itinerary = True
        elif metadata.get("domain") == "multi_domain":
            has_multi_domain = True
        elif metadata.get("domain") == "unknown":
            has_unknown = True

    if has_unknown:
        return {
            "contains_itinerary_state": True,
            "request_mode": "itinerary",
            "safety_class": "unknown",
            "producer_id": SUMMARY_SAFETY_PRODUCER_ID,
            "context_version": SUMMARY_SAFETY_CONTEXT_VERSION,
            "produced_at_utc": produced_at,
        }
    if has_itinerary:
        return {
            "contains_itinerary_state": True,
            "request_mode": "itinerary",
            "safety_class": "itinerary",
            "producer_id": SUMMARY_SAFETY_PRODUCER_ID,
            "context_version": SUMMARY_SAFETY_CONTEXT_VERSION,
            "produced_at_utc": produced_at,
        }
    if has_multi_domain:
        return {
            "contains_itinerary_state": False,
            "request_mode": "multi_domain",
            "safety_class": "standard",
            "producer_id": SUMMARY_SAFETY_PRODUCER_ID,
            "context_version": SUMMARY_SAFETY_CONTEXT_VERSION,
            "produced_at_utc": produced_at,
        }
    return {
        "contains_itinerary_state": False,
        "request_mode": "single_domain",
        "safety_class": "standard",
        "producer_id": SUMMARY_SAFETY_PRODUCER_ID,
        "context_version": SUMMARY_SAFETY_CONTEXT_VERSION,
        "produced_at_utc": produced_at,
    }


def build_confirmation_safety_context(
    active_route: Optional[Dict[str, Any]] = None,
    *,
    domain: Optional[str] = None,
    itinerary_mode: bool = False,
    now_utc: Optional[float] = None,
) -> Dict[str, Any]:
    """Derive ``confirmation_context.v1`` from ``active_route["mode"]`` on the orchestration path.

    Per WP5 specification:
    - Reads ``active_route["mode"]`` using the ``routing_graph.py`` vocabulary:
      ``single_domain``, ``multi_domain``, ``itinerary``, and ``domain_error``.
    - Only ``single_domain`` with ``contains_itinerary_state=False`` is cheap-route eligible;
      every other value, and any absent value, fails closed to the baseline route.
    """
    produced_at = time.time() if now_utc is None else float(now_utc)
    route_dict = active_route if isinstance(active_route, dict) else {}
    raw_mode = route_dict.get("mode")
    route_domain = str(route_dict.get("domain") or domain or "").strip().lower()

    if itinerary_mode or raw_mode == "itinerary" or route_domain == "itinerary":
        return {
            "contains_itinerary_state": True,
            "request_mode": "itinerary",
            "safety_class": "itinerary",
            "producer_id": CONFIRMATION_CONTEXT_PRODUCER_ID,
            "context_version": CONFIRMATION_CONTEXT_VERSION,
            "produced_at_utc": produced_at,
        }

    if raw_mode in _ALLOWED_ROUTE_MODES:
        return {
            "contains_itinerary_state": False,
            "request_mode": str(raw_mode),
            "safety_class": "standard",
            "producer_id": CONFIRMATION_CONTEXT_PRODUCER_ID,
            "context_version": CONFIRMATION_CONTEXT_VERSION,
            "produced_at_utc": produced_at,
        }

    return {
        "contains_itinerary_state": True,
        "request_mode": "multi_domain",
        "safety_class": "unknown",
        "producer_id": CONFIRMATION_CONTEXT_PRODUCER_ID,
        "context_version": CONFIRMATION_CONTEXT_VERSION,
        "produced_at_utc": produced_at,
    }


def infer_caller_policy_metadata() -> Dict[str, Any]:
    """Infer ``policy_tag`` and ``policy_context`` from the call stack when not explicitly provided."""
    inferred: Dict[str, Any] = {}
    frame = inspect.currentframe()
    try:
        curr = frame.f_back if frame is not None else None
        matched_tag: Optional[str] = None
        active_route: Optional[Dict[str, Any]] = None
        domain_val: Optional[str] = None
        summary_messages: Optional[List[Any]] = None
        conv_metadata: Optional[Dict[str, Any]] = None

        while curr is not None:
            fn_name = curr.f_code.co_name
            f_locals = curr.f_locals

            if matched_tag is None and fn_name in CALL_SITE_TAG_MAP:
                matched_tag = CALL_SITE_TAG_MAP[fn_name]

            if matched_tag == POLICY_TAG_CONFIRMATION_PAYLOAD:
                if domain_val is None and isinstance(f_locals.get("domain"), str):
                    domain_val = f_locals["domain"]
                for route_var in ("active_route", "routing", "route"):
                    cand = f_locals.get(route_var)
                    if isinstance(cand, dict) and "mode" in cand:
                        active_route = cand
                        break

            elif matched_tag == POLICY_TAG_CONVERSATION_SUMMARY:
                if summary_messages is None and isinstance(f_locals.get("messages"), list):
                    summary_messages = f_locals["messages"]
                if conv_metadata is None and isinstance(f_locals.get("metadata"), dict):
                    conv_metadata = f_locals["metadata"]

            curr = curr.f_back

        if matched_tag is not None:
            inferred["policy_tag"] = matched_tag
            if matched_tag == POLICY_TAG_CONFIRMATION_PAYLOAD:
                inferred["policy_context"] = build_confirmation_safety_context(
                    active_route, domain=domain_val
                )
            elif matched_tag == POLICY_TAG_CONVERSATION_SUMMARY and summary_messages is not None:
                inferred["policy_context"] = build_summary_safety_context(
                    summary_messages, conversation_metadata=conv_metadata
                )
    finally:
        del frame

    return inferred

