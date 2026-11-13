"""Operational kill switch, circuit breaker, and per-tag enrollment controls for LLM Gateway."""

from __future__ import annotations

import os
import threading
from typing import FrozenSet, Iterable, Optional, Set

from llm_gateway.constants import CHEAP_ROUTE_ELIGIBLE_TAGS, DEFAULT_KILL_SWITCH_REASON


def _default_enrolled_tags() -> Set[str]:
    raw = os.getenv("LLM_GATEWAY_ENROLLED_TAGS")
    if raw is not None:
        return {tag.strip() for tag in raw.split(",") if tag.strip()}
    return set(CHEAP_ROUTE_ELIGIBLE_TAGS)


class KillSwitchManager:
    """Thread-safe kill switch and call-site enrollment state controller."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active: bool = False
        self._reason: Optional[str] = None
        self._enrolled_tags: Set[str] = _default_enrolled_tags()

    def enable(self, reason: str = DEFAULT_KILL_SWITCH_REASON) -> None:
        with self._lock:
            self._active = True
            self._reason = reason

    def disable(self) -> None:
        with self._lock:
            self._active = False
            self._reason = None

    def is_active(self) -> bool:
        with self._lock:
            return self._active

    def get_reason(self) -> Optional[str]:
        with self._lock:
            return self._reason

    def is_tag_enrolled(self, policy_tag: str) -> bool:
        with self._lock:
            return policy_tag in self._enrolled_tags

    def get_enrolled_tags(self) -> FrozenSet[str]:
        with self._lock:
            return frozenset(self._enrolled_tags)

    def set_enrolled_tags(self, tags: Iterable[str]) -> None:
        with self._lock:
            self._enrolled_tags = {str(t).strip() for t in tags if str(t).strip()}

    def enable_tag_enrollment(self, policy_tag: str) -> None:
        with self._lock:
            if policy_tag:
                self._enrolled_tags.add(policy_tag)

    def disable_tag_enrollment(self, policy_tag: str) -> None:
        with self._lock:
            self._enrolled_tags.discard(policy_tag)

    def reset_enrolled_tags(self) -> None:
        with self._lock:
            self._enrolled_tags = _default_enrolled_tags()


_GLOBAL_KILL_SWITCH = KillSwitchManager()


def set_kill_switch(active: bool, reason: str = DEFAULT_KILL_SWITCH_REASON) -> None:
    if active:
        _GLOBAL_KILL_SWITCH.enable(reason)
    else:
        _GLOBAL_KILL_SWITCH.disable()


def is_kill_switch_active() -> bool:
    return _GLOBAL_KILL_SWITCH.is_active()


def get_kill_switch_reason() -> Optional[str]:
    return _GLOBAL_KILL_SWITCH.get_reason()


def is_tag_enrolled(policy_tag: str) -> bool:
    return _GLOBAL_KILL_SWITCH.is_tag_enrolled(policy_tag)


def get_enrolled_tags() -> FrozenSet[str]:
    return _GLOBAL_KILL_SWITCH.get_enrolled_tags()


def set_enrolled_tags(tags: Iterable[str]) -> None:
    _GLOBAL_KILL_SWITCH.set_enrolled_tags(tags)


def enable_tag_enrollment(policy_tag: str) -> None:
    _GLOBAL_KILL_SWITCH.enable_tag_enrollment(policy_tag)


def disable_tag_enrollment(policy_tag: str) -> None:
    _GLOBAL_KILL_SWITCH.disable_tag_enrollment(policy_tag)


def reset_enrolled_tags() -> None:
    _GLOBAL_KILL_SWITCH.reset_enrolled_tags()
