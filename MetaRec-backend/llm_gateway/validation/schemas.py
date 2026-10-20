"""Immutable, versioned Draft-7 schemas for the initial WP5-B call sites.

Registry documents are stored as canonical JSON strings. Resolvers return a
fresh decoded object, so callers cannot mutate the registered definition.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Tuple

from llm_gateway.constants import SemanticValidatorId
from llm_gateway.contracts import ResponseKind

DRAFT7_META_SCHEMA_URI = "http://json-schema.org/draft-07/schema#"
CONVERSATION_SUMMARY_SCHEMA_ID = "conversation.summary.text.v1"
CONFIRMATION_PAYLOAD_SCHEMA_ID = "confirmation.payload.response.v1"

_ALLOWED_PREFERENCE_PATCH_KEYS: Tuple[str, ...] = (
    "amenities",
    "attraction_types",
    "brand",
    "budget",
    "budget_range",
    "category",
    "dietary_restrictions",
    "dining_purpose",
    "exclude_genres",
    "flavor_profiles",
    "genres",
    "location",
    "min_rating",
    "model",
    "mood",
    "product",
    "restaurant_types",
    "stars",
    "subject",
    "tags",
    "typical_budget",
    "use_case",
    "year",
)


def _canonical_schema(schema: Mapping[str, Any]) -> str:
    return json.dumps(
        schema,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


@dataclass(frozen=True)
class ResponseSchemaEntry:
    """Immutable metadata and optional JSON Schema for one response contract."""

    schema_id: str
    kind: ResponseKind
    semantic_validator_id: str
    allow_empty: bool
    _schema_json: Optional[str]

    def schema(self) -> Optional[Dict[str, Any]]:
        """Return an independent copy of the registered schema document."""
        if self._schema_json is None:
            return None
        return json.loads(self._schema_json)


@dataclass(frozen=True)
class ValidatorInputSchemaEntry:
    """Immutable validator-input contract resolved by semantic validator ID."""

    validator_id: str
    _schema_json: str

    def schema(self) -> Dict[str, Any]:
        """Return an independent copy of the registered schema document."""
        return json.loads(self._schema_json)


_CONFIRMATION_PAYLOAD_RESPONSE_SCHEMA: Dict[str, Any] = {
    "$schema": DRAFT7_META_SCHEMA_URI,
    "$id": "urn:metarec:schema:confirmation.payload:response:v1",
    "type": "object",
    "required": ["message"],
    "additionalProperties": False,
    "properties": {
        "message": {"type": "string", "minLength": 1},
        "quick_actions": {
            "type": "array",
            "minItems": 2,
            "maxItems": 4,
            "items": {
                "type": "object",
                "required": ["id", "label", "value", "preference_patch"],
                "additionalProperties": False,
                "properties": {
                    "id": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 66,
                        "pattern": r"^[a-z0-9_\u4e00-\u9fff]+$",
                    },
                    "label": {"type": "string", "minLength": 1, "maxLength": 40},
                    "value": {"type": "string", "minLength": 1},
                    "preference_patch": {
                        "type": "object",
                        "minProperties": 1,
                        "maxProperties": 1,
                        "propertyNames": {"enum": list(_ALLOWED_PREFERENCE_PATCH_KEYS)},
                        "additionalProperties": {},
                    },
                    "message": {"type": "string", "minLength": 1, "maxLength": 120},
                },
            },
        },
    },
}

_NONE_VALIDATOR_INPUT_SCHEMA: Dict[str, Any] = {
    "$schema": DRAFT7_META_SCHEMA_URI,
    "$id": "urn:metarec:schema:conversation.summary:validator-inputs:v1",
    "type": "object",
    "maxProperties": 0,
    "additionalProperties": False,
}

_CONFIRMATION_VALIDATOR_INPUT_SCHEMA: Dict[str, Any] = {
    "$schema": DRAFT7_META_SCHEMA_URI,
    "$id": "urn:metarec:schema:confirmation.payload:validator-inputs:v1",
    "type": "object",
    "required": ["allowed_preference_patch_keys"],
    "additionalProperties": False,
    "properties": {
        "allowed_preference_patch_keys": {
            "type": "array",
            "minItems": 1,
            "maxItems": len(_ALLOWED_PREFERENCE_PATCH_KEYS),
            "uniqueItems": True,
            "items": {"enum": list(_ALLOWED_PREFERENCE_PATCH_KEYS)},
        }
    },
}

_RESPONSE_SCHEMAS: Mapping[str, ResponseSchemaEntry] = MappingProxyType(
    {
        CONVERSATION_SUMMARY_SCHEMA_ID: ResponseSchemaEntry(
            schema_id=CONVERSATION_SUMMARY_SCHEMA_ID,
            kind=ResponseKind.TEXT,
            semantic_validator_id=SemanticValidatorId.NONE.value,
            allow_empty=False,
            _schema_json=None,
        ),
        CONFIRMATION_PAYLOAD_SCHEMA_ID: ResponseSchemaEntry(
            schema_id=CONFIRMATION_PAYLOAD_SCHEMA_ID,
            kind=ResponseKind.JSON,
            semantic_validator_id=SemanticValidatorId.CONFIRMATION_PAYLOAD.value,
            allow_empty=False,
            _schema_json=_canonical_schema(_CONFIRMATION_PAYLOAD_RESPONSE_SCHEMA),
        ),
    }
)

_VALIDATOR_INPUT_SCHEMAS: Mapping[str, ValidatorInputSchemaEntry] = MappingProxyType(
    {
        SemanticValidatorId.NONE.value: ValidatorInputSchemaEntry(
            validator_id=SemanticValidatorId.NONE.value,
            _schema_json=_canonical_schema(_NONE_VALIDATOR_INPUT_SCHEMA),
        ),
        SemanticValidatorId.CONFIRMATION_PAYLOAD.value: ValidatorInputSchemaEntry(
            validator_id=SemanticValidatorId.CONFIRMATION_PAYLOAD.value,
            _schema_json=_canonical_schema(_CONFIRMATION_VALIDATOR_INPUT_SCHEMA),
        ),
    }
)


def resolve_response_schema(schema_id: str) -> Optional[ResponseSchemaEntry]:
    """Resolve a versioned response schema ID, or return ``None`` if unknown."""
    return _RESPONSE_SCHEMAS.get(schema_id)


def resolve_validator_input_schema(validator_id: str) -> Optional[ValidatorInputSchemaEntry]:
    """Resolve a validator's bounded input schema, or ``None`` if unknown."""
    return _VALIDATOR_INPUT_SCHEMAS.get(validator_id)


def registered_response_schema_ids() -> Tuple[str, ...]:
    return tuple(_RESPONSE_SCHEMAS)


def registered_validator_ids() -> Tuple[str, ...]:
    return tuple(_VALIDATOR_INPUT_SCHEMAS)
