from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import yaml


@dataclass
class Config:
    raw: dict[str, Any]

    def __getattr__(self, name: str) -> Any:
        try:
            return self.raw[name]
        except KeyError as e:
            raise AttributeError(name) from e

    def get(self, path: str, default: Any = None) -> Any:
        node: Any = self.raw
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    @classmethod
    def load(cls, path: str | None = None) -> "Config":
        default_path = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs", "default.yaml"))
        cfg_path = path or default_path
        with open(cfg_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls(raw)

    def section(self, name: str) -> dict[str, Any]:
        return self.raw.get(name, {})


def load_config(path: str | None = None) -> Config:
    return Config.load(path)
