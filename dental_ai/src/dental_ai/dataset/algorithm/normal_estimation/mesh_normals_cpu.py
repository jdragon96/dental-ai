from __future__ import annotations

import torch

from .normal_estimation import NormalEstimationInput, NormalEstimationOutput


def mesh_normals_cpu(inp: NormalEstimationInput, out: NormalEstimationOutput) -> bool:
    """면적 가중 정점 normal (segmentation3d mesh_normals 포트).  False: face가 없음.

    cross product의 길이가 face 넓이의 두 배라 raw cross product를 더하면 그게 면적
    가중이다.  방향은 face winding을 따른다 — 스캐너 mesh는 바깥을 향하게 감겨 있다.
    """
    if inp.faces is None or inp.faces.numel() == 0:
        return False
    positions, faces = inp.positions, inp.faces
    tri = positions[faces]                                         # [F,3,3]
    fn = torch.linalg.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    vn = torch.zeros_like(positions)
    for corner in range(3):
        vn.index_add_(0, faces[:, corner], fn)
    out.normals = vn / torch.linalg.vector_norm(vn, dim=1, keepdim=True).clamp_min(1e-12)
    return True
