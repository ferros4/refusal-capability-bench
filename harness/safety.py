"""Shared notices stamped into run metadata and reports."""

from __future__ import annotations

RESEARCH_USE_NOTICE = (
    "Research / evaluation use only. Some datasets contain sensitive or harmful "
    "prompts by design. Do not use outputs to cause harm. Review local law and "
    "organizational policy before sharing result directories."
)

META_SAFETY_FIELDS = {
    "research_use_only": True,
    "research_use_notice": RESEARCH_USE_NOTICE,
    "contains_sensitive_prompts": True,
}
