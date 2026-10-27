from __future__ import annotations

import torch

from .voxel_downsample import VoxelDownsampleInput, VoxelDownsampleOutput


def voxel_grid_cpu(inp: VoxelDownsampleInput, out: VoxelDownsampleOutput) -> bool:
    """voxel마다 centroid에 가장 가까운 정점 하나를 대표로 남긴다.  False: 정점이 없음.

    centroid·차이·거리를 원본(toothseg3d)처럼 float32로, 같은 순서로 계산하고 동률이면
    가장 작은 정점 index를 고른다 — 두 점짜리 voxel은 centroid가 정확히 중점이라
    반올림이 대표를 고르기 때문이다.
    """
    positions = inp.positions
    n_vert = positions.shape[0]
    if n_vert == 0:
        return False
    cell = torch.floor((positions - positions.amin(0)) / inp.voxel_mm).long()
    dims = cell.amax(0) + 1
    key = (cell[:, 0] * dims[1] + cell[:, 1]) * dims[2] + cell[:, 2]
    _, inverse, counts = torch.unique(key, sorted=True, return_inverse=True, return_counts=True)
    n_vox = counts.numel()

    centroid = torch.zeros(n_vox, 3, dtype=positions.dtype)
    centroid.index_add_(0, inverse, positions).div_(counts[:, None].to(positions.dtype))
    d = positions - centroid[inverse]
    dist = (d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1] + d[:, 2] * d[:, 2]).sqrt()
    nearest = torch.full((n_vox,), torch.inf, dtype=dist.dtype)
    nearest.scatter_reduce_(0, inverse, dist, "amin")
    candidate = torch.where(dist == nearest[inverse], torch.arange(n_vert), n_vert)
    representative = torch.full((n_vox,), n_vert, dtype=torch.int64)
    representative.scatter_reduce_(0, inverse, candidate, "amin")
    out.representative = representative
    out.cells = cell[representative]
    return True
