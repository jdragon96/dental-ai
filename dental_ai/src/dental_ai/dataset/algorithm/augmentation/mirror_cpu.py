from __future__ import annotations

import torch

from .augmentation import UNKNOWN_TOOTH, AugmentationInput, AugmentationOutput

# 좌우 반전은 두 사분면을 맞바꾼다: class c in 1..8 <-> c + 8.  17은 위치를 모르니 그대로.
MIRROR_MAP = torch.tensor([0] + list(range(9, 17)) + list(range(1, 9)) + [UNKNOWN_TOOTH])


def mirror_cpu(inp: AugmentationInput, out: AugmentationOutput) -> bool:
    """정중시상면 반사 (x -> -x), 사분면 class도 함께 바뀐다.

    반사는 회전이 아니므로 SO(3) augmentation 위에 더해지는 게 있다 — 치아가 환자의
    어느 쪽에 있는지를 바꾸는 유일한 변환이다.
    """
    s = inp.sample
    flip = torch.tensor([-1.0, 1.0, 1.0])
    out.sample = s.replace(positions=s.positions * flip, normals=s.normals * flip, classes=MIRROR_MAP[s.classes],
                           clean_positions=None if s.clean_positions is None else s.clean_positions * flip)
    return True
