from __future__ import annotations

import math

import torch

from .augmentation import AugmentationInput, AugmentationOutput
from .common.rng import uniform


def wedge_crop_cpu(inp: AugmentationInput, out: AugmentationOutput) -> bool:
    """xy 중심에서 본 악궁의 연속된 부채꼴만 남긴다.  False: 남는 점이 crop_min_points 이하.

    각도를 xy(Teeth3DS 교합면)에서 재므로 회전 전에 돌아야 한다.
    """
    s, p, gen = inp.sample, inp.params, inp.generator
    keep_frac = uniform(gen, *p.wedge_keep)
    start = uniform(gen, -math.pi, math.pi)
    c = s.positions[:, :2] - s.positions[:, :2].mean(0)
    d = torch.remainder(torch.atan2(c[:, 1], c[:, 0]) - start + math.pi, 2 * math.pi) - math.pi
    keep = d.abs() <= math.pi * keep_frac
    if int(keep.sum()) <= p.crop_min_points:
        return False
    out.sample = s.select(keep)
    return True
