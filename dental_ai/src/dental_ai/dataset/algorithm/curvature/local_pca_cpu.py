from __future__ import annotations

import torch

from ..common.spatial import knn_index
from .curvature import CurvatureInput, CurvatureOutput


def local_pca_cpu(inp: CurvatureInput, out: CurvatureOutput) -> bool:
    """점 이웃의 local PCA로 본 표면 모양.  scale(k)마다 두 channel:

    * variation = 3 λ0 / (λ0 + λ1 + λ2): 0 평면 .. 1 등방.  홈·모서리·교두에서 커진다.
    * bend = (이웃 중심 - p) · n / (이웃까지 RMS 거리): 부호 있는 곡률.  + 오목 (치은
      경계, 치간 홈, 교합면 fissure), - 볼록 (교두, 융선).

    둘 다 회전·이동에 불변이라 SO(3) augmentation을 해도 값이 같고, TTA pose마다 다시
    계산할 필요가 없다.  캐시에 face가 없어 mesh 곡률 대신 점 이웃으로 구한다.
    False: 점이 없음.
    """
    n = inp.points.shape[0]
    if n == 0:
        return False
    p = inp.points.float()
    # kNN 검색이 비용의 대부분이다.  가까운 순으로 오므로 가장 큰 k로 한 번만 찾고 잘라 쓴다.
    idx = knn_index(p, max(inp.ks))
    cols = []
    for k in inp.ks:
        d = p[idx[:, :k]] - p[:, None]                                   # [n,k,3] 이웃 - p (자기 자신 포함)
        mean = d.mean(1)
        centred = d - mean[:, None]
        cov = torch.einsum("nki,nkj->nij", centred, centred).double() / d.shape[1]
        ev = torch.linalg.eigvalsh(cov)                                  # 오름차순
        variation = 3 * ev[:, 0].clamp_min(0) / ev.sum(1).clamp_min(1e-12)
        rms = d.square().sum(2).mean(1).sqrt().clamp_min(1e-9)
        bend = (mean * inp.normals).sum(1) / rms
        cols += [variation.float(), bend.clamp(-1.0, 1.0)]
    out.curvature = torch.stack(cols, 1)
    return True
