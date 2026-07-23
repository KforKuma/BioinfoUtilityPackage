from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


_OPERATORS = {
    "<": lambda value, threshold: value < threshold,
    "<=": lambda value, threshold: value <= threshold,
    ">": lambda value, threshold: value > threshold,
    ">=": lambda value, threshold: value >= threshold,
    "==": lambda value, threshold: value == threshold,
    "!=": lambda value, threshold: value != threshold,
}


@dataclass(frozen=True)
class DecisionRule:
    rule_id: str
    metric: str
    operator: str
    threshold: float | bool
    description: str
    method: str = "unspecified"
    effect_component: str = "composition"
    threshold_source: str = "preregistered"
    adjustment_method: str = "not_applicable"
    adjustment_family: str = "not_applicable"
    requires_valid_diagnostics: bool = True
    fallback_enabled: bool = False
    rule_version: str = "v1"
    nominal_alpha: float = 0.05
    adaptive_alpha_enabled: bool = False
    benchmark_enabled: bool = True

    def __post_init__(self) -> None:
        if not self.rule_id:
            raise ValueError("`rule_id` must not be empty.")
        if not self.metric:
            raise ValueError("`metric` must not be empty.")
        if self.operator not in _OPERATORS:
            raise ValueError(f"Unsupported decision operator: {self.operator!r}")
        if self.nominal_alpha != 0.05:
            raise ValueError("Phase 1.5 decision rules require nominal_alpha=0.05.")
        if self.adaptive_alpha_enabled:
            raise ValueError("Phase 1.5 forbids adaptive alpha.")

    def evaluate(self, value: Any) -> bool:
        if value is None:
            raise ValueError("Decision value is unavailable.")
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            raise ValueError("Decision value is unavailable.")
        return bool(_OPERATORS[self.operator](value, self.threshold))


class DecisionRuleRegistry:
    def __init__(self, rules: list[DecisionRule] | tuple[DecisionRule, ...] = ()) -> None:
        self._rules: dict[str, DecisionRule] = {}
        for rule in rules:
            self.register(rule)

    def register(self, rule: DecisionRule) -> None:
        if rule.rule_id in self._rules and self._rules[rule.rule_id] != rule:
            raise ValueError(f"Decision rule ID already exists with different semantics: {rule.rule_id}")
        self._rules[rule.rule_id] = rule

    def get(self, rule_id: str) -> DecisionRule:
        try:
            return self._rules[rule_id]
        except KeyError as exc:
            raise KeyError(f"Unknown decision rule ID: {rule_id!r}") from exc

    def __contains__(self, rule_id: str) -> bool:
        return rule_id in self._rules

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DecisionRuleRegistry":
        source = Path(path)
        with source.open(encoding="utf-8") as handle:
            document = yaml.safe_load(handle)
        records = document.get("decision_rules", []) if isinstance(document, dict) else document
        if not isinstance(records, list) or not records:
            raise ValueError("Decision rule YAML must contain a non-empty `decision_rules` list.")
        rules = []
        for record in records:
            rules.append(DecisionRule(
                rule_id=str(record["decision_rule_id"]),
                metric=str(record["evidence_metric"]),
                operator=str(record["operator"]),
                threshold=record["default_threshold"],
                description=str(record["description"]),
                method=str(record["method"]),
                effect_component=str(record["effect_component"]),
                threshold_source=str(record["threshold_source"]),
                adjustment_method=str(record["adjustment_method"]),
                adjustment_family=str(record["adjustment_family"]),
                requires_valid_diagnostics=bool(record["requires_valid_diagnostics"]),
                fallback_enabled=bool(record["fallback_enabled"]),
                rule_version=str(record["rule_version"]),
                nominal_alpha=float(record.get("nominal_alpha", 0.05)),
                adaptive_alpha_enabled=bool(record.get("adaptive_alpha_enabled", False)),
                benchmark_enabled=bool(record.get("benchmark_enabled", True)),
            ))
        return cls(rules)

    def for_method(self, method: str) -> list[DecisionRule]:
        return [rule for rule in self._rules.values() if rule.method == method]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([
            {
                "decision_rule_id": rule.rule_id,
                "method": rule.method,
                "effect_component": rule.effect_component,
                "evidence_metric": rule.metric,
                "operator": rule.operator,
                "threshold_source": rule.threshold_source,
                "default_threshold": rule.threshold,
                "adjustment_method": rule.adjustment_method,
                "adjustment_family": rule.adjustment_family,
                "requires_valid_diagnostics": rule.requires_valid_diagnostics,
                "fallback_enabled": rule.fallback_enabled,
                "rule_version": rule.rule_version,
                "description": rule.description,
                "nominal_alpha": rule.nominal_alpha,
                "adaptive_alpha_enabled": rule.adaptive_alpha_enabled,
                "benchmark_enabled": rule.benchmark_enabled,
            }
            for rule in self._rules.values()
        ])


def default_decision_rule_path() -> Path:
    return Path(__file__).resolve().parents[3] / "config" / "decision_rules.yaml"


def load_default_decision_rules() -> DecisionRuleRegistry:
    return DecisionRuleRegistry.from_yaml(default_decision_rule_path())
