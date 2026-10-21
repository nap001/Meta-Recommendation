"""Compatibility tests for deterministic confirmation normalization."""

from __future__ import annotations

from copy import deepcopy

import pytest

from llm_gateway.validation import (
    ConfirmationNormalizationContext,
    is_meaningful_patch_value,
    normalize_confirmation_payload,
)

pytestmark = pytest.mark.backend_unit


def _action(
    label: str,
    value: object,
    *,
    action_id: str | None = None,
    patch_key: str = "use_case",
    message: str | None = None,
) -> dict:
    action = {
        "id": action_id,
        "label": label,
        "value": value,
        "preference_patch": {patch_key: value},
    }
    if message is not None:
        action["message"] = message
    return action


def _context(language: str = "en") -> ConfirmationNormalizationContext:
    fallback = "Please confirm this request." if language == "en" else "请确认这个请求。"
    return ConfirmationNormalizationContext(language=language, fallback_message=fallback)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, False),
        ("", False),
        ("   ", False),
        ("ANY", False),
        ([], False),
        ({"nested": []}, False),
        (["", {"nested": "value"}], True),
        (0, True),
        (False, True),
    ],
)
def test_meaningful_patch_value_matches_existing_recursive_rule(value: object, expected: bool) -> None:
    assert is_meaningful_patch_value(value) is expected


def test_normalization_repairs_ids_values_lengths_and_does_not_mutate_input() -> None:
    raw = {
        "message": "Choose a use case.",
        "ignored": "drop me",
        "quick_actions": [
            {
                "label": "L" * 45,
                "value": "",
                "preference_patch": {"use_case": "work"},
                "message": "M" * 125,
                "ignored": True,
            },
            {
                "id": "Work option",
                "label": "Gaming",
                "value": "gaming",
                "preference_patch": {"use_case": "gaming"},
            },
        ],
    }
    before = deepcopy(raw)

    normalized = normalize_confirmation_payload(
        raw,
        allowed_preference_patch_keys=["use_case"],
        context=_context(),
    )

    assert raw == before
    assert set(normalized) == {"message", "quick_actions"}
    first, second = normalized["quick_actions"]
    assert first["id"].startswith("use_case_")
    assert first["value"] == "work"
    assert len(first["label"]) == 40
    assert len(first["message"]) == 120
    assert second["id"] == "work_option"
    assert "ignored" not in first


def test_duplicate_ids_are_repaired_and_duplicate_labels_are_removed_case_insensitively() -> None:
    raw = {
        "message": "Work or gaming?",
        "quick_actions": [
            _action("Work", "work", action_id="choice"),
            _action("WORK", "duplicate", action_id="other"),
            _action("Gaming", "gaming", action_id="choice"),
        ],
    }

    normalized = normalize_confirmation_payload(
        raw,
        allowed_preference_patch_keys=["use_case"],
        context=_context(),
    )

    assert [item["label"] for item in normalized["quick_actions"]] == ["Work", "Gaming"]
    assert [item["id"] for item in normalized["quick_actions"]] == ["choice", "choice_3"]


def test_duplicate_id_repair_avoids_existing_suffixed_ids_and_respects_length_limit() -> None:
    long_id = "x" * 64
    raw = {
        "message": "Choose A, B, C, or D.",
        "quick_actions": [
            _action("A", "a", action_id="choice"),
            _action("B", "b", action_id="choice_3"),
            _action("C", "c", action_id="choice"),
            _action("D", "d", action_id="other"),
        ],
    }

    normalized = normalize_confirmation_payload(
        raw,
        allowed_preference_patch_keys=["use_case"],
        context=_context(),
    )

    ids = [item["id"] for item in normalized["quick_actions"]]
    assert ids[:3] == ["choice", "choice_3", "choice_4"]
    assert len(ids) == len(set(ids))

    long_ids = normalize_confirmation_payload(
        {
            "message": "Choose Long A or Long B.",
            "quick_actions": [
                _action("Long A", "a", action_id=long_id),
                _action("Long B", "b", action_id=long_id),
            ],
        },
        allowed_preference_patch_keys=["use_case"],
        context=_context(),
    )["quick_actions"]
    assert len({item["id"] for item in long_ids}) == 2
    assert all(len(item["id"]) <= 66 for item in long_ids)


def test_labels_are_deduplicated_after_truncation_and_with_unicode_casefold() -> None:
    shared_prefix = "L" * 40
    raw = {
        "message": "Choose an option.",
        "quick_actions": [
            _action(f"{shared_prefix}A", "a"),
            _action(f"{shared_prefix}B", "b"),
            _action("Straße", "street"),
            _action("STRASSE", "street-alt"),
            _action("Gaming", "gaming"),
        ],
    }

    normalized = normalize_confirmation_payload(
        raw,
        allowed_preference_patch_keys=["use_case"],
        context=_context(),
    )

    labels = [item["label"] for item in normalized["quick_actions"]]
    assert labels == [shared_prefix, "Straße", "Gaming"]
    assert len({label.casefold() for label in labels}) == len(labels)


@pytest.mark.parametrize(
    "actions",
    [
        [_action("Only", "one")],
        [_action(str(index), index) for index in range(5)],
        [
            _action("Work", "work", patch_key="use_case"),
            _action("Comedy", "comedy", patch_key="genres"),
        ],
        [
            _action("Work", "work"),
            _action("Unknown", "x", patch_key="unsupported"),
        ],
    ],
)
def test_invalid_action_sets_normalize_to_omission(actions: list[dict]) -> None:
    normalized = normalize_confirmation_payload(
        {"message": "Confirm?", "quick_actions": actions},
        allowed_preference_patch_keys=["use_case", "genres"],
        context=_context(),
    )
    assert normalized == {"message": "Confirm?"}


def test_fallback_message_handles_non_object_and_missing_message() -> None:
    assert normalize_confirmation_payload(
        ["valid JSON", "wrong shape"],
        allowed_preference_patch_keys=["use_case"],
        context=_context(),
    ) == {"message": "Please confirm this request."}
    assert normalize_confirmation_payload(
        {"quick_actions": []},
        allowed_preference_patch_keys=["use_case"],
        context=_context(),
    ) == {"message": "Please confirm this request."}


def test_english_choice_question_is_appended_only_when_no_label_is_mentioned() -> None:
    actions = [_action("Work", "work"), _action("Gaming", "gaming")]
    appended = normalize_confirmation_payload(
        {"message": "Please choose.", "quick_actions": actions},
        allowed_preference_patch_keys=["use_case"],
        context=_context(),
    )
    assert appended["message"] == "Please choose. Would you prefer Work or Gaming?"

    unchanged = normalize_confirmation_payload(
        {"message": "Would Work suit you?", "quick_actions": actions},
        allowed_preference_patch_keys=["use_case"],
        context=_context(),
    )
    assert unchanged["message"] == "Would Work suit you?"


def test_chinese_choice_question_preserves_existing_presentation_behavior() -> None:
    normalized = normalize_confirmation_payload(
        {
            "message": "请选择。",
            "quick_actions": [_action("办公", "work"), _action("游戏", "gaming")],
        },
        allowed_preference_patch_keys=["use_case"],
        context=_context("zh"),
    )
    assert normalized["message"] == "请选择。 您更偏向办公还是游戏呢？"


def test_normalization_context_is_frozen_and_validated() -> None:
    with pytest.raises(ValueError, match="language"):
        ConfirmationNormalizationContext(language="fr", fallback_message="Confirm")
    with pytest.raises(ValueError, match="fallback"):
        ConfirmationNormalizationContext(language="en", fallback_message="  ")
    context = _context()
    with pytest.raises(AttributeError):
        context.language = "zh"  # type: ignore[misc]
