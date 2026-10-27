"""전처리 결과 (ToothSegPreProcessing이 만든 npz 캐시)를 스캔마다 그림 한 장으로 본다.

    dental_ai preprocess-view --cache data/ios/teeth3ds/cache
    dental_ai preprocess-view --cache data/ios/3d_iosseg/cache --name IOSSEG007_upper IOSSEG007_lower
    dental_ai preprocess-view --cache data/ios/3d_iosseg/cache --names data/ios/3d_iosseg/splits/test.txt --count 0

전처리가 올바른지 보는 용도다: 캐시에 든 그대로 (full resolution) 그리고, 색은 colour
augmentation이 칠하는 색을 함께 그린다.  학습은 돌지 않는다.  그림은 --out/<스캔>.png, 3 x 3 패널:

    surface    nrm 음영.  교합면 쪽에서 보이는데 카메라 반대를 향한 normal (뒤집힌 normal)은 빨강
    classes    cls 색 + 치아마다 FDI 번호
    instances  ins 경계선 + 치아마다 id
    colour     colour augmentation (synth_colour_cpu)이 칠하는 색 3 draw.  캐시에 rgb가 없는 스캔은 학습 때
               colour_p 확률로 이렇게 칠해지고, 아니면 색 없이 들어간다.  rgb가 있으면 첫 칸이 실제 색이다
               (학습은 실제 색을 쓰고 칠하지 않는다).  draw는 스캔 이름으로 seed해서 다시 그려도 같다.
    front      정면 (순측)에서 본 cls와 색
    checks     정점·치아 수, 빠진 FDI, class와 instance가 어긋난 치아, normal 점검

교합면 view: 치관 방향은 기하로 (정점 PCA의 가장 얇은 축, 치아가 치은보다 높은 쪽), 화면
위 (전치 쪽)는 좌우가 없는 악궁 위치 번호로 잡는다.  화면 x = 위 × 치관 방향인 진짜 회전이고
사분면 (좌우) label은 view에 쓰지 않으므로, label이 맞으면 위턱은 1x가 왼쪽·2x가 오른쪽,
아래턱은 3x가 왼쪽·4x가 오른쪽에 온다.  좌우가 바뀐 label은 여기서 바로 보인다.

그림은 matplotlib (Agg, pyplot 없이)으로 그려 화면 없는 서버에서도 돈다.
"""
from __future__ import annotations

import argparse
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

# augmentation.UNKNOWN_TOOTH와 같은 캐시 class 값.  거기서 import하면 torch가 모듈 맨 위에서 딸려
# 온다 — torch는 colour_augmentation()에서만, 순서를 지켜 올린다 (거기 설명).
UNKNOWN_TOOTH = 17
COLOUR_DRAWS = 3                 # 그릴 colour augmentation draw 수

# ------------------------------------------------------------------ colours
SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
# 악궁을 따라 18..11 | 21..28: 첫 사분면은 위치 순서대로, 둘째 사분면은 거꾸로 slot을 받아서
# 이웃한 두 치아 (중절치 쌍 포함)는 언제나 서로 다른 slot이다.  치아마다 FDI 번호도 붙는다.
SLOTS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
GINGIVA = "#d6d5ce"
UNKNOWN = "#898781"              # class 17: 위치 모르는 치아
FLIPPED = "#d03b3b"              # 카메라 반대를 향한 normal
TOOTH_FILL, GINGIVA_FILL = "#d6d5ce", "#efeee9"
EDGE = "#52514e"                 # instance 경계


def _rgb(h: str) -> np.ndarray:
    return np.array([int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)])


CLASS_RGB = np.array([_rgb(h) for h in (GINGIVA, *SLOTS, *SLOTS[::-1], UNKNOWN)])          # [18,3]
PANEL_PX = 640                   # 패널 하나를 그리는 해상도
# 교합면에서 보이는 픽셀 중 카메라 반대를 향한 normal의 비율이 이보다 크면 문제로 표시한다.  성긴
# 스캔은 점 사이로 반대편 벽이 비쳐 보여서 정상이어도 0이 아니다: 정상 캐시 최대 5.2% (Teeth3DS),
# 9.4% (3D-IOSSeg), normal을 전부 뒤집으면 최소 90.6%.
BACK_FACING_LIMIT = 0.25


# ------------------------------------------------------------------ labels
def class_to_fdi(c: int, jaw_upper: bool) -> str:
    """class 0..17 -> FDI 번호 (ToothSegPreProcessing.fdi_to_class의 역)."""
    if c == 0:
        return "gingiva"
    if c == UNKNOWN_TOOTH:
        return "?"
    first, second = (1, 2) if jaw_upper else (3, 4)
    return f"{first if c <= 8 else second}{(c - 1) % 8 + 1}"


def arch_position(classes: np.ndarray) -> np.ndarray:
    """class -> 좌우 없는 악궁 위치 (0 gingiva / 모름, 1 중절치 .. 8 제3대구치)."""
    tooth = (classes >= 1) & (classes <= 16)
    return np.where(tooth, (classes.astype(np.int64) - 1) % 8 + 1, 0)


# -------------------------------------------------------------------- scan
@dataclass
class CachedScan:
    name: str
    path: Path
    xyz: np.ndarray              # [V,3] float64 mm
    normals: np.ndarray          # [V,3] float64, unit length
    classes: np.ndarray          # [V] int64
    instances: np.ndarray        # [V] int64
    jaw_upper: bool
    rgb: np.ndarray | None       # [V,3] float64 0..1
    zero_normals: int            # 길이 0인 normal 수 (면 없는 정점)


def load_cache(path: Path) -> CachedScan:
    with np.load(path) as d:
        nrm = d["nrm"].astype(np.float64)
        length = np.linalg.norm(nrm, axis=1)
        return CachedScan(path.stem, path, d["xyz"].astype(np.float64), nrm / np.maximum(length, 1e-9)[:, None],
                          d["cls"].astype(np.int64), d["ins"].astype(np.int64), bool(d["jaw"]),
                          d["rgb"].astype(np.float64) / 255.0 if "rgb" in d.files else None,
                          int((length < 1e-3).sum()))


# -------------------------------------------------------------------- view
def view_frames(scan: CachedScan) -> tuple[np.ndarray, np.ndarray]:
    """(교합면, 정면) [3,3] 회전.  행이 화면 x (오른쪽), y (위), z (카메라 쪽)."""
    c = scan.xyz - scan.xyz.mean(0)
    pca = np.linalg.svd(c, full_matrices=False)[2]
    tooth, gingiva = scan.classes > 0, scan.classes == 0
    crown = pca[2]
    # 치아가 치은보다 높은 쪽이 치관 방향.  치은이 없으면 치아 normal 평균으로 정한다.
    if tooth.any() and gingiva.any():
        lean = (c[tooth].mean(0) - c[gingiva].mean(0)) @ crown
    else:
        lean = scan.normals[tooth].sum(0) @ crown
    crown = crown if lean >= 0 else -crown

    # 위치 번호 (1 중절치 .. 8 제3대구치)가 작아지는 쪽 = 전치 쪽
    pos = arch_position(scan.classes)
    t = pos > 0
    if t.any() and np.ptp(pos[t]) > 0:
        anterior = -((pos[t] - pos[t].mean())[:, None] * (c[t] - c[t].mean(0))).sum(0)
    else:
        anterior = pca[0]
    anterior = anterior - (anterior @ crown) * crown
    if np.linalg.norm(anterior) < 1e-6:
        anterior = pca[1] - (pca[1] @ crown) * crown
    anterior /= np.linalg.norm(anterior)

    occlusal = np.stack([np.cross(anterior, crown), anterior, crown])
    up = -crown if scan.jaw_upper else crown                     # 정면: 해부학적 위.  위턱은 치관이 아래를 향한다
    front = np.stack([np.cross(up, anterior), up, anterior])
    return occlusal, front


@dataclass
class Render:
    """z-buffer 한 장: 픽셀마다 보이는 정점 번호 (-1 배경)와 화면 좌표 (mm)."""

    index: np.ndarray            # [H,W] int64
    q: np.ndarray                # [V,3] 화면 좌표 (x 오른쪽, y 위, z 카메라 쪽)
    extent: tuple[float, float, float, float]

    def colour(self, rgb: np.ndarray) -> np.ndarray:
        """[V,3] 정점 색 -> [H,W,3] 그림 (배경은 SURFACE)."""
        img = np.empty((*self.index.shape, 3))
        img[:] = _rgb(SURFACE)
        hit = self.index >= 0
        img[hit] = rgb[self.index[hit]]
        return img

    def visible(self) -> np.ndarray:
        """보이는 정점 번호 (픽셀마다 하나, 중복 있음)."""
        return self.index[self.index >= 0]


def render(xyz: np.ndarray, frame: np.ndarray, size: int = PANEL_PX) -> Render:
    """정점을 (2r+1)² 칸으로 찍고 픽셀마다 카메라에 가장 가까운 정점만 남긴다.  r은 정점 간격에 맞춘다
    (3D-IOSSeg는 Teeth3DS보다 10배 성기다)."""
    q = (xyz - xyz.mean(0)) @ frame.T
    lo, hi = q[:, :2].min(0), q[:, :2].max(0)
    half = 0.53 * float((hi - lo).max())
    centre = (lo + hi) / 2
    px = size / (2 * half)                                       # mm -> px
    u = np.floor((q[:, 0] - centre[0] + half) * px).astype(np.int64)
    v = np.floor((centre[1] + half - q[:, 1]) * px).astype(np.int64)
    spacing = np.sqrt(float(np.prod(hi - lo)) * 0.4 / len(q)) * px
    radius = int(np.clip(round(0.6 * spacing), 1, 5))

    r = range(-radius, radius + 1)
    uu = np.concatenate([u + dx for dy in r for dx in r])
    vv = np.concatenate([v + dy for dy in r for dx in r])
    ids = np.tile(np.arange(len(q)), len(r) ** 2)
    ok = (uu >= 0) & (uu < size) & (vv >= 0) & (vv < size)
    pix, ids = vv[ok] * size + uu[ok], ids[ok]
    order = np.lexsort((-q[ids, 2], pix))                        # 픽셀별로, 그 안에서 카메라에 가까운 순
    first = np.r_[True, pix[order][1:] != pix[order][:-1]]
    index = np.full(size * size, -1, np.int64)
    index[pix[order][first]] = ids[order][first]
    extent = (centre[0] - half, centre[0] + half, centre[1] - half, centre[1] + half)
    return Render(index.reshape(size, size), q, extent)


def shading(normals: np.ndarray, frame: np.ndarray) -> np.ndarray:
    """[V,1] 카메라 쪽 조명.  뒤집힌 normal도 밝기는 같게 (|n·z|)."""
    return (0.62 + 0.38 * np.abs(normals @ frame[2]))[:, None]


def instance_edges(index: np.ndarray, instances: np.ndarray) -> np.ndarray:
    """[H,W] bool: 이웃 픽셀과 instance가 다른 픽셀 (배경과의 경계는 뺀다)."""
    ins = np.where(index >= 0, instances[np.maximum(index, 0)], -1)
    across = (ins[:, 1:] != ins[:, :-1]) & (ins[:, 1:] >= 0) & (ins[:, :-1] >= 0)
    down = (ins[1:] != ins[:-1]) & (ins[1:] >= 0) & (ins[:-1] >= 0)
    edge = np.zeros(ins.shape, bool)
    edge[:, 1:] |= across
    edge[:, :-1] |= across
    edge[1:] |= down
    edge[:-1] |= down
    return edge


# ------------------------------------------------------------------ checks
@dataclass
class Tooth:
    instance: int
    points: int
    cls: int                     # 다수결 class
    stray: int                   # 다수결과 다른 class인 정점 수


def teeth_of(scan: CachedScan) -> list[Tooth]:
    out = []
    for i in np.unique(scan.instances):
        if i == 0:
            continue
        counts = np.bincount(scan.classes[scan.instances == i], minlength=UNKNOWN_TOOTH + 1)
        out.append(Tooth(int(i), int(counts.sum()), int(counts.argmax()), int(counts.sum() - counts.max())))
    return out


def check_lines(scan: CachedScan, teeth: list[Tooth], back_facing: float) -> list[tuple[str, bool]]:
    """checks 패널의 (글, 문제인가) 줄들."""
    def fdi(c: int) -> str:
        return class_to_fdi(c, scan.jaw_upper)

    v = len(scan.xyz)
    lines = [(f"vertices      {v:,}", False),
             (f"extent        {' x '.join(f'{e:.1f}' for e in np.ptp(scan.xyz, 0))} mm", False),
             (f"gingiva       {(scan.classes == 0).mean():.1%} of vertices", False),
             (f"colour        {'rgb (real)' if scan.rgb is not None else 'none: colour augmentation paints it'}",
              False),
             ("", False)]

    # 치아 / class
    present = sorted({t.cls for t in teeth if 1 <= t.cls <= 16})
    lines.append((f"teeth         {len(teeth)} instances", False))
    first, second = (1, 2) if scan.jaw_upper else (3, 4)
    for quadrant, classes in ((first, range(1, 9)), (second, range(9, 17))):
        have = " ".join(fdi(c) if c in present else "--" for c in classes)
        lines.append((f"  Q{quadrant}          {have}", False))
    unknown = [t for t in teeth if t.cls == UNKNOWN_TOOTH]
    if unknown:
        lines.append((f"unknown pos.  {len(unknown)} instances (class {UNKNOWN_TOOTH})", False))
    if teeth:
        small = min(teeth, key=lambda t: t.points)
        lines.append((f"smallest      {fdi(small.cls)} (id {small.instance}): {small.points:,} vertices",
                      small.points < 0.002 * v))
    lines.append(("", False))

    # class와 instance가 서로 맞는가
    mixed = [t for t in teeth if t.stray]
    lines.append((f"mixed teeth   {len(mixed)}  (instance with > 1 class)", bool(mixed)))
    for t in mixed[:3]:
        lines.append((f"  id {t.instance}: {t.stray:,} of {t.points:,} vertices not {fdi(t.cls)}", True))
    shared = [c for c in present if sum(t.cls == c for t in teeth) > 1]
    lines.append((f"split classes {len(shared)}  (class over > 1 instance)"
                   + ("" if not shared else ": " + " ".join(fdi(c) for c in shared)), bool(shared)))
    gum_class = int(((scan.instances == 0) & (scan.classes > 0)).sum())
    lines.append((f"gingiva ins.  {gum_class:,} vertices with a tooth class", gum_class > 0))
    lines.append(("", False))

    # normal
    # face 없는 외톨이 정점 몇 개 (Teeth3DS 70개 스캔에 1~3개씩)는 흔하고 무해하다
    lines.append((f"zero normals  {scan.zero_normals:,}", scan.zero_normals > 0.001 * v))
    lines.append((f"back-facing   {back_facing:.1%} of visible occlusal pixels", back_facing > BACK_FACING_LIMIT))
    return lines


# ----------------------------------------------------- colour augmentation
def colour_augmentation(scan: CachedScan, draws: int = COLOUR_DRAWS) -> tuple[list[np.ndarray], float]:
    """학습의 colour augmentation (synth_colour_cpu)이 이 스캔에 칠하는 색 draw들 [V,3] 0..1과, 학습에서
    그 단계가 도는 확률 colour_p.  draw k는 '<스캔 이름>/k'로 seed한다.

    torch는 여기서, numpy가 OpenMP를 먼저 띄운 뒤에 올린다.  conda의 MKL numpy와 pip torch는 OpenMP
    runtime (libiomp5md.dll)을 따로 들고 있어서, torch가 먼저 올라온 프로세스는 첫 numpy 행렬 연산에서
    죽는다 (OMP: Error #15).  numpy가 먼저면 둘이 같이 돈다 (pip numpy에서는 아무 일도 하지 않는다).
    """
    np.linalg.svd(np.eye(3))                     # MKL의 OpenMP를 torch보다 먼저 띄운다
    import torch

    from .algorithm.augmentation.augmentation import (AugmentationInput, AugmentationOutput, AugmentationParams,
                                                      ScanSample)
    from .algorithm.augmentation.synth_colour_cpu import synth_colour_cpu

    torch.set_num_threads(1)                     # worker가 여럿이니 worker마다 1 thread
    sample = ScanSample(torch.from_numpy(scan.xyz.astype(np.float32)),
                        torch.from_numpy(scan.normals.astype(np.float32)),
                        torch.from_numpy(scan.classes), torch.from_numpy(scan.instances))
    params = AugmentationParams()
    colours = []
    for k in range(draws):
        out = AugmentationOutput()
        gen = torch.Generator().manual_seed(zlib.crc32(f"{scan.name}/{k}".encode()))
        if not synth_colour_cpu(AugmentationInput(sample, params, gen), out):
            raise ValueError("synth_colour_cpu did not paint")
        colours.append(out.sample.colour.double().numpy())
    return colours, params.colour_p


# ------------------------------------------------------------------ figure
def _panel(ax, img: np.ndarray, extent: tuple[float, float, float, float], title: str) -> None:
    ax.imshow(img, extent=extent, interpolation="antialiased")
    ax.set_title(title, loc="left", fontsize=10, color=INK, pad=6)
    ax.set_axis_off()


def _label(ax, x: float, y: float, s: str, size: float = 7.5) -> None:
    ax.text(x, y, s, ha="center", va="center", fontsize=size, color=INK,
            bbox=dict(boxstyle="round,pad=0.18", fc=SURFACE, ec="none", alpha=0.85))


def figure(scan: CachedScan) -> Figure:
    occ_frame, front_frame = view_frames(scan)
    occ, front = render(scan.xyz, occ_frame), render(scan.xyz, front_frame)
    shade = shading(scan.normals, occ_frame)
    teeth = teeth_of(scan)
    centre = {t.instance: occ.q[scan.instances == t.instance, :2].mean(0) for t in teeth}

    fig = Figure(figsize=(15, 15.6), dpi=110, facecolor=SURFACE)
    axes = fig.subplots(3, 3, gridspec_kw=dict(wspace=0.04, hspace=0.1, left=0.01, right=0.99, top=0.945,
                                                bottom=0.01))

    # surface: 뒤집힌 normal은 빨강
    facing = scan.normals @ occ_frame[2]
    vis = occ.visible()
    back = float((facing[vis] < 0).mean()) if len(vis) else 0.0
    grey = _rgb(SURFACE) * (0.45 + 0.5 * np.clip(facing, 0, 1))[:, None]
    surface = np.where((facing < 0)[:, None], _rgb(FLIPPED), grey)
    ax = axes[0, 0]
    _panel(ax, occ.colour(surface), occ.extent, f"surface: normals (red = facing away, {back:.1%})")
    x0, _, y0, _ = occ.extent
    span = occ.extent[1] - x0
    ax.plot([x0 + 0.04 * span, x0 + 0.04 * span + 10], [y0 + 0.04 * span] * 2, color=INK2, lw=2,
            solid_capstyle="butt")
    ax.text(x0 + 0.04 * span + 5, y0 + 0.06 * span, "10 mm", ha="center", va="bottom", fontsize=8, color=INK2)

    # classes: FDI 번호
    ax = axes[0, 1]
    left, right = ("1x", "2x") if scan.jaw_upper else ("3x", "4x")
    _panel(ax, occ.colour(CLASS_RGB[scan.classes] * shade), occ.extent,
           f"classes: FDI (correct labels: {left} left, {right} right)")
    for t in teeth:
        _label(ax, *centre[t.instance], class_to_fdi(t.cls, scan.jaw_upper))

    # instances: 경계선 + id
    ax = axes[0, 2]
    fill = np.where((scan.instances > 0)[:, None], _rgb(TOOTH_FILL), _rgb(GINGIVA_FILL)) * shade
    img = occ.colour(fill)
    img[instance_edges(occ.index, scan.instances)] = _rgb(EDGE)
    _panel(ax, img, occ.extent, f"instances: {len(teeth)} teeth (id)")
    for t in teeth:
        _label(ax, *centre[t.instance], str(t.instance))

    # colour: 학습의 colour augmentation이 칠하는 색.  실제 색이 있으면 첫 칸은 실제 색이다.
    synth, colour_p = colour_augmentation(scan)
    if scan.rgb is not None:
        colours = [scan.rgb, *synth[:COLOUR_DRAWS - 1]]
        names = ["real rgb", *[f"synth draw {k + 1}" for k in range(COLOUR_DRAWS - 1)]]
        notes = [f"training uses it with p={colour_p:g}, else none", *["not used: scan has rgb"] * (COLOUR_DRAWS - 1)]
    else:
        colours = synth
        names = [f"colour augmentation draw {k + 1}" for k in range(COLOUR_DRAWS)]
        notes = [f"training: p={colour_p:g}, else none", *[""] * (COLOUR_DRAWS - 1)]
    for ax, colour, name, note in zip(axes[1], colours, names, notes):
        _panel(ax, occ.colour(colour), occ.extent, f"{name} ({note})" if note else name)

    # front: 해부학적 정면 (환자를 마주 본다)
    front_shade = shading(scan.normals, front_frame)
    _panel(axes[2, 0], front.colour(CLASS_RGB[scan.classes] * front_shade), front.extent, "front view: classes")
    _panel(axes[2, 1], front.colour(colours[0]), front.extent, f"front view: {names[0]}")

    # checks
    ax = axes[2, 2]
    ax.set_axis_off()
    ax.set_title("checks", loc="left", fontsize=10, color=INK, pad=6)
    for k, (s, problem) in enumerate(check_lines(scan, teeth, back)):
        ax.text(0.02, 0.97 - k * 0.048, ("! " if problem else "  ") + s, transform=ax.transAxes, va="top",
                family="monospace", fontsize=8.5, color=FLIPPED if problem else INK)

    fig.suptitle(f"{scan.name}    {'upper' if scan.jaw_upper else 'lower'} jaw    {len(scan.xyz):,} vertices    "
                 f"{scan.path}", x=0.01, ha="left", fontsize=11, color=INK)
    return fig


def save_view(path: Path, out_dir: Path) -> tuple[Path, str | None]:
    """캐시 스캔 하나 -> out_dir/<스캔>.png.  (그림 경로, 오류)."""
    dst = out_dir / f"{path.stem}.png"
    try:
        figure(load_cache(path)).savefig(dst, facecolor=SURFACE)
        return dst, None
    except Exception as e:                       # 깨진 캐시 하나가 전체를 멈추면 안 된다
        return dst, repr(e)


def _run(job: tuple[Path, Path]) -> tuple[Path, str | None]:
    return save_view(*job)


# -------------------------------------------------------------------- main
def find_scans(cache_dirs: list[Path], names: list[str] | None) -> tuple[list[Path], list[str]]:
    """(찾은 npz 경로, 못 찾은 이름).  이름은 그걸 가진 첫 캐시에서 읽는다 (ToothSegDataset과 같다)."""
    if names is None:
        names = sorted({p.stem for d in cache_dirs for p in d.glob("*.npz")})
    found, missing = [], []
    for n in names:
        hit = next((d / f"{n}.npz" for d in cache_dirs if (d / f"{n}.npz").exists()), None)
        if hit:
            found.append(hit)
        else:
            missing.append(n)
    return found, missing


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, nargs="+", required=True, help="npz 캐시 폴더 (여러 개 가능)")
    ap.add_argument("--name", nargs="+", default=None, help="볼 스캔 이름 (예: IOSSEG007_upper)")
    ap.add_argument("--names", type=Path, default=None, help="스캔 이름 목록 파일 (예: <cache>/test.txt)")
    ap.add_argument("--start", type=int, default=0, help="처음 볼 스캔 번호")
    ap.add_argument("--count", type=int, default=16, help="그릴 스캔 수 (0이면 전부)")
    ap.add_argument("--out", type=Path, default=Path("runs/preprocess_view"))
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    names = a.name or ([ln.strip() for ln in a.names.read_text().splitlines() if ln.strip()] if a.names else None)
    scans, missing = find_scans(a.cache, names)
    for n in missing:
        print(f"  missing {n}")
    scans = scans[a.start:a.start + a.count] if a.count else scans[a.start:]
    if not scans:
        raise SystemExit(f"no scans to draw in {[str(d) for d in a.cache]}")
    a.out.mkdir(parents=True, exist_ok=True)
    print(f"{len(scans)} scans -> {a.out}", flush=True)

    t0, errs = time.time(), []
    jobs = [(p, a.out) for p in scans]
    # Pool과 달리 worker가 죽으면 기다리지 않고 BrokenProcessPool로 멈춘다
    with ProcessPoolExecutor(max(1, min(a.workers, len(jobs)))) as pool:
        for dst, err in pool.map(_run, jobs, chunksize=4):
            if err:
                errs.append((dst, err))
                print(f"  error {dst.stem} {err}", flush=True)
            else:
                print(f"  {dst}", flush=True)
    print(f"done in {time.time() - t0:.1f} s, {len(errs)} errors")


if __name__ == "__main__":
    main()
