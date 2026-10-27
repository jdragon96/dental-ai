"""CPU torch.Generator 하나에서 뽑는 random draw.  gen=None이면 torch 전역 RNG.

구현마다 뽑는 순서가 segmentation3d와 같아야 같은 seed에서 같은 augmentation이 나온다.
"""
from __future__ import annotations

import torch


def uniform(gen: torch.Generator | None, lo: float, hi: float) -> float:
    return lo + (hi - lo) * float(torch.rand((), generator=gen, dtype=torch.float64))


def randint(gen: torch.Generator | None, lo: int, hi: int) -> int:
    """[lo, hi) uniform 정수."""
    return int(torch.randint(lo, hi, (), generator=gen))


def chance(gen: torch.Generator | None, p: float) -> bool:
    return p > 0 and uniform(gen, 0.0, 1.0) < p


def choice(gen: torch.Generator | None, weights: tuple[float, ...]) -> int:
    return int(torch.multinomial(torch.tensor(weights, dtype=torch.float64), 1, generator=gen))


def normal(gen: torch.Generator | None, shape: tuple[int, ...], std: float,
           dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.randn(shape, generator=gen, dtype=dtype) * std
