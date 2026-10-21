"""Member B validation foundations for the versioned LLM gateway contracts."""

from llm_gateway.validation.preflight import (
    PreflightIssue,
    PreflightIssueCode,
    ValidationPreflightResult,
    preflight_response_contract,
    schemas_structurally_equal,
    validate_draft7_schema_definition,
)
from llm_gateway.validation.schemas import (
    CONFIRMATION_PAYLOAD_SCHEMA_ID,
    CONVERSATION_SUMMARY_SCHEMA_ID,
    DRAFT7_META_SCHEMA_URI,
    ResponseSchemaEntry,
    ValidatorInputSchemaEntry,
    registered_response_schema_ids,
    registered_validator_ids,
    resolve_response_schema,
    resolve_validator_input_schema,
)
from llm_gateway.validation.hook import (
    ExternalValidationHook,
    create_wp5_external_validation_hook,
)
from llm_gateway.validation.normalization import (
    ConfirmationNormalizationContext,
    is_meaningful_patch_value,
    normalize_confirmation_payload,
)
from llm_gateway.validation.semantic import (
    is_complete_confirmation_validator_registered,
    register_wp5_semantic_validators,
    validate_confirmation_payload_semantics,
)

__all__ = [
    "CONFIRMATION_PAYLOAD_SCHEMA_ID",
    "CONVERSATION_SUMMARY_SCHEMA_ID",
    "DRAFT7_META_SCHEMA_URI",
    "ConfirmationNormalizationContext",
    "ExternalValidationHook",
    "PreflightIssue",
    "PreflightIssueCode",
    "ResponseSchemaEntry",
    "ValidationPreflightResult",
    "ValidatorInputSchemaEntry",
    "preflight_response_contract",
    "registered_response_schema_ids",
    "registered_validator_ids",
    "create_wp5_external_validation_hook",
    "is_complete_confirmation_validator_registered",
    "is_meaningful_patch_value",
    "normalize_confirmation_payload",
    "register_wp5_semantic_validators",
    "resolve_response_schema",
    "resolve_validator_input_schema",
    "schemas_structurally_equal",
    "validate_draft7_schema_definition",
    "validate_confirmation_payload_semantics",
]
