"""
Reading config.yaml.

One file holds the settings worth changing, and both the tool that creates a
model and the one that trains it read it, so a model cannot be built with one
shape and trained with another. Command-line flags still win where they are
given - the file is the default, not a cage.
"""

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT = os.path.join(ROOT, "config.yaml")


def load(path=None):
    import yaml
    path = path or DEFAULT
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def get(cfg, dotted, fallback=None):
    """get(cfg, 'pool.d_ff') - missing sections return the fallback."""
    cur = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return fallback
        cur = cur[part]
    return cur
