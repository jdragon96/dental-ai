from __future__ import annotations

import torch

from ..common.spatial import nearest_distance
from .augmentation import CROP_MODES, AugmentationInput, AugmentationOutput
from .common.rng import choice, randint, uniform

MIN_PICKED_POINTS = 200          # 고른 치아가 이보다 작으면 그 주변을 crop하지 않는다


def _tooth_centroids(positions: torch.Tensor, instances: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """[G,3] float64, `ids`(정렬됨)의 치아마다 centroid."""
    tooth = instances > 0
    slot = torch.searchsorted(ids, instances[tooth])
    acc = torch.zeros(ids.numel(), 3, dtype=torch.float64)
    acc.index_add_(0, slot, positions[tooth].double())
    return acc / torch.bincount(slot, minlength=ids.numel())[:, None]


def _arch_order(centroids: torch.Tensor) -> list[int]:
    """악궁 한쪽 끝에서 반대쪽 끝까지, 가장 가까운 이웃을 따라 걷는다.

    중심각으로 정렬하면 atan2의 branch cut이 악궁 어딘가를 갈라 연속 구간이 둘로
    끊긴다.  가장 먼 끝에서 사슬을 이으면 좌표계 가정이 없다.
    """
    d = torch.cdist(centroids, centroids)
    n = d.shape[0]
    first = int(torch.argmax(d)) // n                    # 가장 긴 구간의 한쪽 끝
    dl = d.tolist()
    order, left = [first], set(range(n)) - {first}
    while left:
        nxt = min(left, key=lambda j: dl[order[-1]][j])
        order.append(nxt)
        left.discard(nxt)
    return order


def segment_crop_cpu(inp: AugmentationInput, out: AugmentationOutput) -> bool:
    """연속된 치아 몇 개와 그 주변 gingiva만 남긴다 — 실제 부분 스캔의 모양.

    bridge prep은 이웃한 치아 2~5개와 gingiva 띠, quadrant는 악궁의 1/4~1/2, half는 절반.
    악궁을 따라 치아를 고르고 그 바깥으로 margin을 키운다.  False: 치아가 3개 미만,
    고른 치아가 너무 작음, 또는 남는 점이 crop_min_points 이하.
    """
    s, p, gen = inp.sample, inp.params, inp.generator
    mode = p.crop_mode or CROP_MODES[choice(gen, p.crop_mix)]
    positions, instances = s.positions, s.instances
    ids = torch.unique(instances)
    ids = ids[ids > 0]
    n = ids.numel()
    if n < 3:
        return False
    order = _arch_order(_tooth_centroids(positions, instances, ids))
    if mode == "bridge":
        k = randint(gen, 2, min(6, n + 1))
    elif mode == "quadrant":
        k = max(2, round(n * uniform(gen, 0.25, 0.5)))
    else:                                                # half
        k = max(3, round(n * uniform(gen, 0.45, 0.6)))
    # 악궁은 고리가 아니라 열린 U자다: 마지막 대구치를 넘어 감으면 입 안쪽을 건너뛰어
    # 끊어진 두 조각이 남는다
    k = min(k, n)
    start = randint(gen, 0, n - k + 1)
    pick = ids[torch.tensor(order[start:start + k])]

    sel = torch.isin(instances, pick)
    if int(sel.sum()) < MIN_PICKED_POINTS:
        return False
    margin = uniform(gen, *p.crop_margin_mm[mode])
    keep = nearest_distance(positions, positions[sel]) < margin

    # 치아는 통째로 들어가거나 빠진다 — 라벨 붙일 조각이 남지 않게.  고른 치아는
    # 아래 `| sel`이 다시 통째로 넣는다.
    tooth = instances > 0
    slot = torch.searchsorted(ids, instances[tooth])
    kept = torch.zeros(n, dtype=torch.float64)
    kept.index_add_(0, slot, keep[tooth].double())
    keep[tooth] = (kept / torch.bincount(slot, minlength=n) > 0.5)[slot]
    keep |= sel
    if int(keep.sum()) <= p.crop_min_points:
        return False
    out.sample = s.select(keep)
    return True
