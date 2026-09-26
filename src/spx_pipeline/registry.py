"""Dataset registry: parses ``registry.yaml`` into typed ``DatasetConfig`` objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_REGISTRY = Path(__file__).with_name("registry.yaml")


@dataclass(frozen=True)
class CleaningRule:
    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetConfig:
    name: str
    description: str
    source_format: str
    source_path: str
    store_path: str
    partition_cols: list[str]
    granularity: str
    symbol: str
    schema: dict[str, str]
    drop_cols: list[str] = field(default_factory=list)
    cleaning_rules: list[CleaningRule] = field(default_factory=list)
    index_col: str = "timestamp"
    rows_per_day: int = 0
    compression: str = "zstd"
    compression_level: int = 3
    row_group_size: int = 300_000
    timezone: str = "America/New_York"

    @property
    def is_bulk(self) -> bool:
        """Bulk sources (a ``*`` in the path) are streamed file by file in batches."""
        return "*" in self.source_path

    @property
    def is_single_file(self) -> bool:
        return not self.partition_cols and "{date}" not in self.source_path


_SOURCE_FORMATS = {"csv", "parquet"}


class Registry:
    """Load and access dataset configurations from ``registry.yaml``.

    Names listed under ``aliases`` resolve to their canonical dataset, so research code
    written against an older name keeps working.
    """

    def __init__(self, yaml_path: Path | str):
        self._path = Path(yaml_path)
        with open(self._path) as f:
            raw = yaml.safe_load(f) or {}

        self._version = raw.get("version", 1)
        defaults = raw.get("defaults", {})
        self._datasets: dict[str, DatasetConfig] = {}

        for name, cfg in (raw.get("datasets") or {}).items():
            source_format = cfg["source_format"]
            if source_format not in _SOURCE_FORMATS:
                raise ValueError(f"{name}: unknown source_format {source_format!r}")
            self._datasets[name] = DatasetConfig(
                name=name,
                description=cfg.get("description", ""),
                source_format=source_format,
                source_path=cfg["source_path"],
                store_path=cfg["store_path"],
                partition_cols=list(cfg.get("partition_cols", [])),
                granularity=cfg.get("granularity", defaults.get("granularity", "1min")),
                symbol=cfg.get("symbol", ""),
                schema=dict(cfg.get("schema", {})),
                drop_cols=list(cfg.get("drop_cols", [])),
                cleaning_rules=_parse_cleaning_rules(cfg.get("cleaning_rules", [])),
                index_col=cfg.get("index_col", "timestamp"),
                rows_per_day=cfg.get("rows_per_day", 0),
                compression=cfg.get("compression", defaults.get("compression", "zstd")),
                compression_level=cfg.get(
                    "compression_level", defaults.get("compression_level", 3)
                ),
                row_group_size=cfg.get("row_group_size", defaults.get("row_group_size", 300_000)),
                timezone=cfg.get("timezone", defaults.get("timezone", "America/New_York")),
            )

        self._aliases: dict[str, str] = dict(raw.get("aliases") or {})
        for alias, target in self._aliases.items():
            if target not in self._datasets:
                raise ValueError(f"alias {alias!r} points to unknown dataset {target!r}")
            if alias in self._datasets:
                raise ValueError(f"alias {alias!r} shadows a dataset of the same name")

    @property
    def version(self) -> int:
        return int(self._version)

    def resolve(self, name: str) -> str:
        """Canonical dataset name for ``name`` (itself, or the target of an alias)."""
        if name in self._datasets:
            return name
        if name in self._aliases:
            return self._aliases[name]
        raise KeyError(f"Dataset '{name}' not in registry. Available: {self.list_datasets()}")

    def get(self, name: str) -> DatasetConfig:
        return self._datasets[self.resolve(name)]

    def list_datasets(self) -> list[str]:
        return list(self._datasets)

    @property
    def aliases(self) -> dict[str, str]:
        return dict(self._aliases)

    def __contains__(self, name: object) -> bool:
        return name in self._datasets or name in self._aliases

    def __repr__(self) -> str:
        return f"Registry(datasets={self.list_datasets()})"


def _parse_cleaning_rules(raw_rules: list[Any]) -> list[CleaningRule]:
    rules = []
    for rule in raw_rules or []:
        if isinstance(rule, dict):
            rule_name, rule_params = next(iter(rule.items()))
            if isinstance(rule_params, bool):
                rule_params = {"enabled": rule_params}
            elif rule_params is None:
                rule_params = {}
            elif not isinstance(rule_params, dict):
                rule_params = {"value": rule_params}
            rules.append(CleaningRule(name=rule_name, params=rule_params))
        elif isinstance(rule, str):
            rules.append(CleaningRule(name=rule))
        else:
            raise ValueError(f"Cannot parse cleaning rule: {rule!r}")
    return rules
