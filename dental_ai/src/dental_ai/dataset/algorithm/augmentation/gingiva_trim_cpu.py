from __future__ import annotations

import torch

from ..common.spatial import nearest_distance
from .augmentation import AugmentationInput, AugmentationOutput
from .common.rng import uniform


def gingiva_trim_cpu(inp: AugmentationInput, out: AugmentationOutput) -> bool:
    """가장 가까운 치아에서 reach_mm보다 먼 gingiva를 버린다.

    Teeth3DS는 구개와 넓은 gingiva를 남긴다 — 치아 밖으로 11.9 mm(p95)까지.  in-house
    스캐너는 4.6 mm로 다듬는다.  이게 없으면 model은 실제로 돌릴 스캔 모양을 못 본다.
    False: 치아나 gingiva가 없음, 또는 남는 점이 trim_min_points 이하.
    """
    s, p, gen = inp.sample, inp.params, inp.generator
    reach = uniform(gen, *p.trim_reach_mm)
    tooth = s.instances != 0
    if not bool(tooth.any()) or bool(tooth.all()):
        return False
    gum = ~tooth
    keep = torch.ones_like(tooth)
    keep[gum] = nearest_distance(s.positions[gum], s.positions[tooth]) <= reach
    if int(keep.sum()) <= p.trim_min_points:
        return False
    out.sample = s.select(keep)
    return True
