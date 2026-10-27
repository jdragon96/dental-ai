"""cKDTree 이웃 검색 — augmentation(crop, gingiva trim)과 curvature가 함께 쓴다."""
from __future__ import annotations

import torch
from scipy.spatial import cKDTree
from torch.utils.data import get_worker_info


def kdtree_workers() -> int:
    """DataLoader worker 안에서는 1 thread (worker가 이미 여럿), 메인 프로세스에서는 전부."""
    return 1 if get_worker_info() is not None else -1


def nearest_distance(query: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """query [Q,3]의 각 점에서 가장 가까운 ref [R,3] 점까지 거리 -> [Q] float32."""
    if ref.shape[0] == 0:
        return torch.full((query.shape[0],), torch.inf)
    dist, _ = cKDTree(ref.numpy()).query(query.numpy(), workers=kdtree_workers())
    return torch.from_numpy(dist).float()


def knn_index(points: torch.Tensor, k: int) -> torch.Tensor:
    """[n,3] float32의 각 점에서 가까운 순으로 이웃 index -> [n, min(k, n)] (자기 자신 포함)."""
    n = points.shape[0]
    p = points.numpy()
    _, idx = cKDTree(p).query(p, k=min(k, n), workers=kdtree_workers())
    return torch.from_numpy(idx.reshape(n, -1))
