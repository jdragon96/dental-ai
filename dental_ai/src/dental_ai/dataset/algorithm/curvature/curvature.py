"""곡률 feature의 Input / Output."""
from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class CurvatureInput:
    points: torch.Tensor             # [n,3] float32, mm
    normals: torch.Tensor            # [n,3] unit length, 바깥 방향 — bend의 부호가 여기에 달렸다
    ks: tuple[int, ...] = (32,)      # 이웃 수 (scale).  0.25 mm voxel에서 k=32 ≈ 반경 0.8 mm


@dataclass
class CurvatureOutput:
    # [n, 2 * len(ks)]: scale마다 (variation, bend)
    curvature: torch.Tensor = field(default_factory=lambda: torch.empty(0, 2))
