"""Resolve which Track (1 or 2) a config model name belongs to, and its input_size."""
from __future__ import annotations


def resolve_track(cfg: dict, model_name: str) -> str | None:
    tracks = cfg.get("tracks", {})
    for track_name in ("track1", "track2"):
        if model_name in tracks.get(track_name, {}).get("backbones", []):
            return track_name
    return None


def track_input_size(cfg: dict, model_name: str) -> int | None:
    track_name = resolve_track(cfg, model_name)
    if track_name is None:
        return None
    return cfg.get("tracks", {}).get(track_name, {}).get("input_size")
