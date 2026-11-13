"""Route determination and ladder construction for LLM Gateway Engine."""

from __future__ import annotations

import logging
import time
from typing import List, Optional, Tuple

from llm_gateway.constants import (
    CHEAP_ROUTE_ELIGIBLE_TAGS,
    DEFAULT_CONTEXT_MAX_AGE_SECONDS,
    DEFAULT_CONTEXT_MAX_FUTURE_SKEW_SECONDS,
    NEVER_CHEAP_ROUTE_TAGS,
    REGISTERED_CONTEXT_PRODUCERS,
    REGISTERED_CONTEXT_VERSIONS,
)
from llm_gateway.contracts import (
    LLMRequestV1,
    PolicyContextV1,
    PolicyLeg,
    RequestMode,
    RouteV1,
    RoutingMode,
)
from llm_gateway.kill_switch import (
    get_kill_switch_reason,
    is_kill_switch_active,
    is_tag_enrolled,
)

logger = logging.getLogger(__name__)

# Low-risk tags eligible for cheap routing when safe
_CHEAP_ROUTE_ELIGIBLE_TAGS = CHEAP_ROUTE_ELIGIBLE_TAGS


def is_policy_context_fresh_and_valid(
    ctx: PolicyContextV1,
    *,
    now_utc: Optional[float] = None,
    max_age_seconds: float = DEFAULT_CONTEXT_MAX_AGE_SECONDS,
) -> bool:
    """Verify producer registration, context version, safety class, and freshness TTL."""
    if ctx.safety_class == "unknown":
        return False
    if ctx.producer_id not in REGISTERED_CONTEXT_PRODUCERS:
        return False
    if ctx.context_version not in REGISTERED_CONTEXT_VERSIONS:
        return False
    current_utc = time.time() if now_utc is None else now_utc
    age = current_utc - ctx.produced_at_utc
    if age < -DEFAULT_CONTEXT_MAX_FUTURE_SKEW_SECONDS or age > max_age_seconds:
        return False
    return True


def determine_initial_route(
    request: LLMRequestV1,
    *,
    now_utc: Optional[float] = None,
) -> Tuple[RouteV1, PolicyLeg]:
    """Determine whether request starts on cheap_route or baseline_route."""
    routing = request.routing_policy
    if is_kill_switch_active():
        logger.info("Kill switch active (%s); forcing baseline route", get_kill_switch_reason())
        return routing.baseline_route, PolicyLeg.BASELINE

    if routing.mode == RoutingMode.CHEAP_FIRST and routing.cheap_route is not None:
        if routing.allowed_provider_ids and routing.cheap_route.provider_id not in routing.allowed_provider_ids:
            return routing.baseline_route, PolicyLeg.BASELINE
        ctx = request.policy_context
        tag = request.policy_tag
        if tag in NEVER_CHEAP_ROUTE_TAGS:
            return routing.baseline_route, PolicyLeg.BASELINE
        is_tag_eligible = (tag in _CHEAP_ROUTE_ELIGIBLE_TAGS) and is_tag_enrolled(tag)
        is_safety_ok = (
            (not ctx.contains_itinerary_state)
            and (ctx.request_mode == RequestMode.SINGLE_DOMAIN)
            and is_policy_context_fresh_and_valid(ctx, now_utc=now_utc)
        )
        if is_tag_eligible and is_safety_ok:
            return routing.cheap_route, PolicyLeg.CHEAP

    return routing.baseline_route, PolicyLeg.BASELINE


def build_route_ladder(
    request: LLMRequestV1, initial_route: RouteV1, initial_leg: PolicyLeg
) -> List[Tuple[RouteV1, PolicyLeg]]:
    """Build the ordered list of routes (route, policy_leg) to attempt."""
    routing = request.routing_policy
    ladder: List[Tuple[RouteV1, PolicyLeg]] = [(initial_route, initial_leg)]

    if initial_leg == PolicyLeg.CHEAP:
        # After cheap route, always fall back to baseline
        ladder.append((routing.baseline_route, PolicyLeg.BASELINE))

    for fb in routing.fallback_routes:
        ladder.append((fb, PolicyLeg.FALLBACK))

    return ladder
