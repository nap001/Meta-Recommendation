"""Route determination and ladder construction for LLM Gateway Engine."""

from __future__ import annotations

import logging
from typing import List, Tuple

from llm_gateway.constants import CHEAP_ROUTE_ELIGIBLE_TAGS
from llm_gateway.contracts import (
    LLMRequestV1,
    PolicyLeg,
    RequestMode,
    RouteV1,
    RoutingMode,
)
from llm_gateway.kill_switch import get_kill_switch_reason, is_kill_switch_active

logger = logging.getLogger(__name__)

# Low-risk tags eligible for cheap routing when safe
_CHEAP_ROUTE_ELIGIBLE_TAGS = CHEAP_ROUTE_ELIGIBLE_TAGS


def determine_initial_route(request: LLMRequestV1) -> Tuple[RouteV1, PolicyLeg]:
    """Determine whether request starts on cheap_route or baseline_route."""
    routing = request.routing_policy
    if is_kill_switch_active():
        logger.info("Kill switch active (%s); forcing baseline route", get_kill_switch_reason())
        return routing.baseline_route, PolicyLeg.BASELINE

    if routing.mode == RoutingMode.CHEAP_FIRST and routing.cheap_route is not None:
        # Check safety allowlist
        ctx = request.policy_context
        tag = request.policy_tag
        is_tag_eligible = tag in _CHEAP_ROUTE_ELIGIBLE_TAGS
        is_safety_ok = (not ctx.contains_itinerary_state) and (ctx.request_mode == RequestMode.SINGLE_DOMAIN)
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
