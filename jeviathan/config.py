"""Profile-based configuration for Jeviathan.

A profile binds a machine (GPU, model, serving stack) to backend + sampling
settings. Select with the JEVIATHAN_PROFILE env var (default: laptop-4050).
JEVIATHAN_BASE_URL / JEVIATHAN_MODEL override the backend at runtime.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
PROFILES_DIR = REPO_ROOT / "profiles"


@dataclass
class BackendConfig:
    type: str = "openai_compatible"
    base_url: str = "http://localhost:11434/v1"
    model: str = "llama3.1:8b"
    api_key: str | None = None
    timeout_s: float = 120.0
    max_retries: int = 3


@dataclass
class SamplingConfig:
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 2048


@dataclass
class CalibrationConfig:
    enabled: bool = False
    path: str | None = None


@dataclass
class LimitsConfig:
    max_questions: int = 64
    max_options_per_choice: int = 255
    max_state_chars: int = 32000


@dataclass
class Profile:
    name: str
    backend: BackendConfig = field(default_factory=BackendConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    limits: LimitsConfig = field(default_factory=LimitsConfig)


def _build(data: dict, name: str) -> Profile:
    return Profile(
        name=name,
        backend=BackendConfig(**dict(data.get("backend") or {})),
        sampling=SamplingConfig(**dict(data.get("sampling") or {})),
        calibration=CalibrationConfig(**dict(data.get("calibration") or {})),
        limits=LimitsConfig(**dict(data.get("limits") or {})),
    )


def load_profile(name: str | None = None) -> Profile:
    name = name or os.environ.get("JEVIATHAN_PROFILE") or "laptop-4050"
    as_path = Path(name)
    path = as_path if as_path.suffix in (".yaml", ".yml") else PROFILES_DIR / f"{name}.yaml"
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    profile = _build(data, name)
    if os.environ.get("JEVIATHAN_BASE_URL"):
        profile.backend.base_url = os.environ["JEVIATHAN_BASE_URL"]
    if os.environ.get("JEVIATHAN_MODEL"):
        profile.backend.model = os.environ["JEVIATHAN_MODEL"]
    return profile


def active_profile() -> Profile:
    return load_profile(None)
