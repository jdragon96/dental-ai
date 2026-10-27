"""voxel downsample의 Input / Output."""
from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class VoxelDownsampleInput:
    positions: torch.Tensor          # [V,3] float32, mm
    voxel_mm: float


@dataclass
class VoxelDownsampleOutput:
    # [N] voxel마다 대표 정점 index, voxel key 오름차순
    representative: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.int64))
    # [N,3] 대표 정점이 속한 voxel의 정수 좌표
    cells: torch.Tensor = field(default_factory=lambda: torch.empty(0, 3, dtype=torch.int64))
