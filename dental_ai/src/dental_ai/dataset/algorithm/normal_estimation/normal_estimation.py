"""정점 normal 추정의 Input / Output."""
from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class NormalEstimationInput:
    positions: torch.Tensor                  # [V,3] float32
    faces: torch.Tensor | None = None        # [F,3] int64


@dataclass
class NormalEstimationOutput:
    normals: torch.Tensor = field(default_factory=lambda: torch.empty(0, 3))    # [V,3] float32, unit length
