"""
Loads config.yaml into a simple nested-attribute object so the rest of the
code can write `cfg.model.fusion_dim` instead of `cfg["model"]["fusion_dim"]`.
"""
from __future__ import annotations
import yaml
from pathlib import Path


class AttrDict(dict):
    """dict that also supports attribute access, recursively."""

    def __getattr__(self, key):
        try:
            val = self[key]
        except KeyError as e:
            raise AttributeError(key) from e
        if isinstance(val, dict):
            val = AttrDict(val)
            self[key] = val
        return val

    def __setattr__(self, key, value):
        self[key] = value


def load_config(path: str | Path = "config.yaml") -> AttrDict:
    path = Path(path)
    if not path.exists():
        # allow running scripts from inside scripts/ or notebooks/
        candidates = [Path.cwd() / "config.yaml", Path.cwd().parent / "config.yaml"]
        for c in candidates:
            if c.exists():
                path = c
                break
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return AttrDict(raw)


def project_root(cfg_path: str | Path = "config.yaml") -> Path:
    """Resolve project root as the directory that contains config.yaml."""
    path = Path(cfg_path)
    if path.exists():
        return path.resolve().parent
    for c in [Path.cwd(), Path.cwd().parent]:
        if (c / "config.yaml").exists():
            return c
    return Path.cwd()
