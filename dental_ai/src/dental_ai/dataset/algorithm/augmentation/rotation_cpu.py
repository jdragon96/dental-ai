from __future__ import annotations

import math

import torch

from .augmentation import AugmentationInput, AugmentationOutput
from .common.rng import normal, uniform


def random_so3(gen: torch.Generator | None) -> torch.Tensor:
    """SO(3)에서 uniform하게 뽑은 회전 (Shoemake quaternion), [3,3] float64.

    스캐너가 쓰는 좌표계와 무관해지게 한다: Teeth3DS는 교합면을 xy에, in-house 스캐너는 xz에 둔다.
    """
    u1, u2, u3 = torch.rand(3, generator=gen, dtype=torch.float64).tolist()
    a, b = math.sqrt(1 - u1), math.sqrt(u1)
    x, y = a * math.sin(2 * math.pi * u2), a * math.cos(2 * math.pi * u2)
    z, w = b * math.sin(2 * math.pi * u3), b * math.cos(2 * math.pi * u3)
    return torch.tensor([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]], dtype=torch.float64)


def small_rotation(gen: torch.Generator | None, rot_xy_deg: float, rot_z_deg: float) -> torch.Tensor:
    """Rx @ Ry @ Rz: 작은 기울기와 z축으로 더 큰 회전, [3,3] float64."""
    ax = math.radians(uniform(gen, -rot_xy_deg, rot_xy_deg))
    ay = math.radians(uniform(gen, -rot_xy_deg, rot_xy_deg))
    az = math.radians(uniform(gen, -rot_z_deg, rot_z_deg))
    cx, cy, cz = math.cos(ax), math.cos(ay), math.cos(az)
    sx, sy, sz = math.sin(ax), math.sin(ay), math.sin(az)
    rx = torch.tensor([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=torch.float64)
    ry = torch.tensor([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=torch.float64)
    rz = torch.tensor([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=torch.float64)
    return rx @ ry @ rz


def rotation_cpu(inp: AugmentationInput, out: AugmentationOutput) -> bool:
    """params.rotation의 회전 ("so3" / "limited") + scale + jitter, float64로.

    이동은 없다 — downsample 뒤 원점 중심으로 다시 옮기므로 이동은 곧바로 지워진다.
    jitter 전 위치는 clean_positions로 남긴다: 곡률은 jitter 없는 표면에서 잰다.
    """
    s, p, gen = inp.sample, inp.params, inp.generator
    r = random_so3(gen) if p.rotation == "so3" else small_rotation(gen, p.rot_xy_deg, p.rot_z_deg)
    scale = uniform(gen, 1 - p.scale, 1 + p.scale)
    pos = (s.positions.double() @ r.T) * scale
    clean = pos if s.clean_positions is None else (s.clean_positions.double() @ r.T) * scale
    if p.jitter_mm > 0:
        pos = pos + normal(gen, tuple(pos.shape), p.jitter_mm, torch.float64)
    nrm = s.normals.double() @ r.T                       # normal도 같은 행렬로
    jittered = p.jitter_mm > 0 or s.clean_positions is not None
    out.sample = s.replace(positions=pos.float(), normals=nrm.float(),
                           clean_positions=clean.float() if jittered else None)
    return True
