"""Tool catalog scoping, parameter projection, and request normalization for LLM Gateway."""

from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple

from llm_gateway.constants import DEFAULT_TOOL_NAME_PATTERN
from llm_gateway.contracts import (
    LLMRequestV1,
    LLMToolDefinitionV1,
    LLMToolProjectionV1,
    ResponseKind,
    RouteV1,
)
from llm_gateway.registry import ProviderRegistry

logger = logging.getLogger(__name__)

_TOOL_NAME_ALLOWED_RE = re.compile(DEFAULT_TOOL_NAME_PATTERN)


def clamp_output_tokens(request: LLMRequestV1, route: RouteV1, registry: ProviderRegistry) -> int:
    """Clamp requested max_output_tokens to model's registered maximum limit.

    Fix #10: Pre-validate and clamp with a WARNING so caller is informed instead of wasting an attempt on 400.
    """
    requested = request.generation.max_output_tokens
    provider_cap = registry.get_provider(route.provider_id)
    if provider_cap:
        for m in provider_cap.models:
            if m.model_id == route.model_id:
                limit = m.max_output_tokens
                if requested > limit:
                    logger.warning(
                        "Request %s asks for %d output tokens but model %s/%s limit is %d; "
                        "clamping to avoid a guaranteed 400 error (Fix #10).",
                        request.request_id,
                        requested,
                        route.provider_id,
                        route.model_id,
                        limit,
                    )
                    return limit
                break
    return requested


def build_tool_projection(
    request: LLMRequestV1, route: RouteV1
) -> Tuple[Optional[LLMToolProjectionV1], Dict[str, str]]:
    """Build internal LLMToolProjectionV1 enforcing monotonic WP7 tool catalog scoping (WP5 L226-234)."""
    if not request.tools or not request.tools.tools:
        return None, {}
    excluded_names = set(request.tools.excluded) if request.tools.excluded else set()
    taken = {t.name for t in request.tools.tools}
    sanitized_mapping: Dict[str, str] = {}
    definitions: List[LLMToolDefinitionV1] = []

    for t in request.tools.tools:
        orig_name = t.name
        if orig_name in excluded_names:
            continue
        wire_name = orig_name
        if not _TOOL_NAME_ALLOWED_RE.match(orig_name):
            candidate = re.sub(r"[^a-zA-Z0-9_-]", "_", orig_name)
            while candidate in taken or candidate in sanitized_mapping:
                candidate += "_"
            wire_name = candidate
            sanitized_mapping[wire_name] = orig_name

        definitions.append(
            LLMToolDefinitionV1(
                source_tool_name=orig_name,
                llm_tool_name=wire_name,
                description=t.description,
                parameters=t.parameters or {"type": "object", "properties": {}},
            )
        )

    projection = LLMToolProjectionV1(
        projection_id=f"proj-{uuid.uuid4().hex[:12]}",
        catalog_id=request.tools.catalog_id,
        health_snapshot_id=request.tools.health_snapshot_id,
        provider_id=route.provider_id,
        definitions=definitions,
    )
    return projection, sanitized_mapping


def project_prompt_embedded_tools(
    request: LLMRequestV1,
    messages: List[Dict[str, Any]],
    projection: Optional[LLMToolProjectionV1],
) -> List[Dict[str, Any]]:
    """Enforce WP7 tool catalog scoping on prompt-embedded tool definitions (Two Tool Shapes)."""
    if not request.tools:
        return messages

    excluded_names = set(request.tools.excluded) if request.tools.excluded else set()
    allowed_source_names = (
        {d.source_tool_name for d in projection.definitions}
        if projection
        else ({t.name for t in request.tools.tools} - excluded_names)
    )

    projected_messages: List[Dict[str, Any]] = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, str):
            projected_messages.append(msg)
            continue

        modified = False
        # 1. JSON payload with "available_tools" (propose_gather_action shape)
        if '"available_tools"' in content:
            try:
                parsed = json.loads(content)
                if isinstance(parsed, dict) and isinstance(parsed.get("available_tools"), dict):
                    filtered_tools = {
                        k: v
                        for k, v in parsed["available_tools"].items()
                        if k in allowed_source_names and k not in excluded_names
                    }
                    parsed["available_tools"] = filtered_tools
                    content = json.dumps(parsed, ensure_ascii=False)
                    modified = True
            except (json.JSONDecodeError, TypeError):
                pass

        # 2. ReAct text prompt containing [可用工具] or 【可用工具】
        if "[可用工具]" in content or "【可用工具】" in content:
            lines = content.splitlines()
            projected_lines: List[str] = []
            in_tools_block = False
            for line in lines:
                stripped = line.strip()
                if "[可用工具]" in stripped or "【可用工具】" in stripped:
                    in_tools_block = True
                    projected_lines.append(line)
                    continue
                if in_tools_block and stripped.startswith("[") and stripped.endswith("]"):
                    in_tools_block = False
                if in_tools_block and stripped.startswith("【") and stripped.endswith("】"):
                    in_tools_block = False

                if in_tools_block:
                    tool_bullet_match = re.match(r"^\s*\d+[\)\.]\s*([a-zA-Z0-9_\.-]+)", line)
                    if tool_bullet_match:
                        t_name = tool_bullet_match.group(1).rstrip(":")
                        if t_name in excluded_names or (allowed_source_names and t_name not in allowed_source_names):
                            modified = True
                            continue
                    elif any(ex in line for ex in excluded_names):
                        modified = True
                        continue
                projected_lines.append(line)
            if modified:
                content = "\n".join(projected_lines)

        if modified:
            new_msg = dict(msg)
            new_msg["content"] = content
            projected_messages.append(new_msg)
        else:
            projected_messages.append(msg)

    return projected_messages


def normalize_request_kwargs(
    request: LLMRequestV1,
    route: RouteV1,
    adaptations: Dict[str, Any],
    registry: ProviderRegistry,
) -> Tuple[Dict[str, Any], Dict[str, str], Optional[LLMToolProjectionV1]]:
    """Convert LLMRequestV1 into provider SDK kwargs with wire sanitization."""
    messages_wire = []
    for msg in request.messages:
        m_dict: Dict[str, Any] = {"role": msg.role.value, "content": msg.content}
        if msg.name:
            m_dict["name"] = msg.name
        messages_wire.append(m_dict)

    if adaptations.get("validation_repair"):
        errors = "; ".join(adaptations.get("validation_errors", []))
        messages_wire.append({
            "role": "user",
            "content": f"The previous response failed contract validation: {errors}. Please return a repaired output strictly conforming to the required schema and constraints."
        })

    # Build projection and wire tools (Two tool shapes supported)
    projection, sanitized_mapping = build_tool_projection(request, route)

    # Project prompt-embedded tools in messages (Two Tool Shapes architecture)
    messages_wire = project_prompt_embedded_tools(request, messages_wire, projection)

    kwargs: Dict[str, Any] = {
        "model": route.model_id,
        "messages": messages_wire,
    }

    gen = request.generation
    if gen.temperature is not None and "temperature" not in adaptations.get("removed_params", []):
        kwargs["temperature"] = gen.temperature
    if gen.seed is not None and "seed" not in adaptations.get("removed_params", []):
        kwargs["seed"] = gen.seed
    if gen.stop:
        kwargs["stop"] = gen.stop

    # Fix #10: use clamped token value
    effective_max_tokens = clamp_output_tokens(request, route, registry)

    # Reasoning model check or max tokens adaptation
    provider_cap = registry.get_provider(route.provider_id)
    is_reasoning = False
    if provider_cap:
        for m in provider_cap.models:
            if m.model_id == route.model_id and m.is_reasoning_model:
                is_reasoning = True
                break

    if is_reasoning or "use_max_completion_tokens" in adaptations:
        kwargs.pop("temperature", None)
        kwargs["extra_body"] = {"max_completion_tokens": effective_max_tokens}
    else:
        kwargs["max_tokens"] = effective_max_tokens

    # JSON response format
    if request.response_contract.kind == ResponseKind.JSON:
        if provider_cap and provider_cap.features.supports_json_object:
            kwargs["response_format"] = {"type": "json_object"}

    # Native tools wire argument
    if projection is not None:
        if provider_cap is None or provider_cap.features.supports_tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": d.llm_tool_name,
                        "description": d.description,
                        "parameters": d.parameters,
                    },
                }
                for d in projection.definitions
            ]

        if request.tool_choice:
            if isinstance(request.tool_choice, str):
                kwargs["tool_choice"] = request.tool_choice
            elif isinstance(request.tool_choice, dict):
                choice_func = request.tool_choice.get("function", {})
                fn_name = choice_func.get("name", "")
                reverse_map = {d.source_tool_name: d.llm_tool_name for d in projection.definitions}
                wire_choice_name = reverse_map.get(fn_name, fn_name)
                kwargs["tool_choice"] = {
                    "type": "function",
                    "function": {"name": wire_choice_name},
                }

    return kwargs, sanitized_mapping, projection
