"""IOS 오픈 데이터셋 원본 -> data/ios/<데이터셋>/raw의 label 내장 PLY.

    dental_ai import-raw teeth3ds        --src data/open/teeth3ds/raw  --out data/ios/teeth3ds
    dental_ai import-raw 3d_iosseg       --src data/open/3d_iosseg     --out data/ios/3d_iosseg
    dental_ai import-raw poseidon3d      --src data/open/poseidon3d/data --out data/ios/poseidon3d
    dental_ai import-raw cbct_ios_fusion --src data/open/cbct_ios_fusion_liu2023/IOSWithGT \
                                         --out data/ios/cbct_ios_fusion_liu2023
    (= python -m dental_ai.dataset.ToothSegRawImport <인자>.  원본은 data/ios/<데이터셋>/_zips에서 다시 푼다)

data/ios/<데이터셋>/
    raw/<이름>_{upper,lower}.ply    binary PLY: 정점 xyz [+ rgb], face, 정점 label (int fdi, int label,
                                    int instance — DentalLabeler와 같은 필드)
    cache/<이름>_{upper,lower}.npz  dental_ai preprocess가 raw에서 만든다 (ToothSegPreProcessing)
    splits/<split>.txt              데이터셋 공식 split (raw / cache 이름 목록)

    fdi       치아 번호 11..48, 위치가 없으면 0
    label     정점 클래스 (VertexLabel): -1 미분류, 0 치은, 1 치아, 2 스캔바디 ...
    instance  치아 instance 번호, 치아가 아니면 0

label은 데이터셋마다 형식이 달라 여기서 정점 FDI로 통일하고, 범주는 FDI에서 정한다 (치은 / 치아):
    teeth3ds          정점 FDI + instance (json).  label 없는 3DTeethLand 테스트 스캔은 label 없이
    3d_iosseg         face 색 -> FDI (IOSSEG_FDI, 색표를 기하에서 추정) -> 정점 다수결
    poseidon3d        face 영역 (pkl) -> 슬롯 (metadata.json) -> FDI (poseidon_slot_fdi) -> 정점 다수결
    cbct_ios_fusion   face FDI (txt) -> 정점 다수결
instance는 Teeth3DS만 원본 id를 쓰고, 나머지는 치아(FDI)마다 하나다.
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
import time
from collections.abc import Callable
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from .ToothSegPreProcessing import real_colour
from .algorithm.scan_decoder.obj_decoder_cpu import obj_decoder_cpu
from .algorithm.scan_decoder.ply_decoder_cpu import ply_decoder_cpu
from .algorithm.scan_decoder.scan_decoder import ScanDecoderInput, ScanDecoderOutput, VertexLabel
from .algorithm.scan_decoder.stl_decoder_cpu import stl_decoder_cpu
from .algorithm.scan_encoder.ply_encoder_cpu import ply_encoder_cpu
from .algorithm.scan_encoder.scan_encoder import ScanEncoderInput, ScanEncoderOutput

DATASETS = ("teeth3ds", "3d_iosseg", "poseidon3d", "cbct_ios_fusion")

# ---------------------------------------------------------------- 3D-IOSSeg
# 3D-IOSSeg는 label을 face 색으로 저장하고, 색 -> FDI 표는 공개돼 있지 않다.  181개 스캔에서
# 추정했다: 스캔마다 +x가 앞(전치), +y가 교합 방향(치관 쪽, gingiva는 -y)이고, 색마다 턱,
# 좌우(z 부호), 앞뒤 순서(x)가 모든 스캔에서 같다.  환자 오른쪽 = anterior × superior이고
# superior는 위턱 -y, 아래턱 +y이므로 위턱은 z < 0, 아래턱은 z > 0이 환자 오른쪽이다.
IOSSEG_FDI = {
    # 위턱 환자 오른쪽 (z < 0): 중절치 .. 제3대구치
    (255, 0, 0): 11, (0, 0, 255): 12, (0, 255, 0): 13, (255, 255, 0): 14,
    (0, 255, 255): 15, (255, 0, 255): 16, (30, 144, 255): 17, (0, 255, 127): 18,
    # 위턱 환자 왼쪽 (z > 0)
    (127, 255, 0): 21, (255, 246, 143): 22, (60, 179, 113): 23, (255, 106, 106): 24,
    (131, 111, 255): 25, (255, 211, 155): 26, (255, 99, 71): 27, (155, 48, 255): 28,
    # 아래턱 환자 왼쪽 (z < 0)
    (255, 48, 48): 31, (0, 191, 255): 32, (255, 165, 0): 33, (202, 255, 112): 34,
    (200, 255, 255): 35, (255, 228, 255): 36, (255, 155, 255): 37, (255, 69, 0): 38,
    # 아래턱 환자 오른쪽 (z > 0)
    (139, 0, 0): 41, (144, 238, 144): 42, (0, 139, 139): 43, (0, 0, 139): 44,
    (139, 0, 139): 45, (255, 105, 180): 46, (230, 230, 250): 47, (255, 228, 181): 48,
}
IOSSEG_GINGIVA = (255, 255, 255)
# 팔레트 색에서 이만큼(RGB 거리) 안이면 그 색의 변형으로 본다 — (139, 0, 2), (255, 165, 2) 등
IOSSEG_COLOUR_TOLERANCE = 40.0
# 원본 이름만 받는다: test/003_L_color.ply, _white.ply는 같은 mesh의 시각화본이다
IOSSEG_NAME = re.compile(r"(\d{3})_([UL])\.ply")

# --------------------------------------------------------------- Poseidon3D
# face 값 17은 치은, 0..N-1은 있는 치아.  metadata.json의 <jaw>_region_mapping[slot] = 영역 값이고
# 슬롯 0은 늘 -1, 슬롯 1..16은 악궁 한쪽 끝에서 다른 끝까지다 (번호가 이웃한 슬롯은 198개
# 스캔 모두에서 공간에서도 이웃).  슬롯 1..8은 198개 중 196개에서 환자 왼쪽이었다 (환자
# 오른쪽 = anterior × superior, Teeth3DS 전문가 label에서 99.4% 맞는 규칙) — 데이터셋의
# L8..L1, R1..R8 순서다.
POSEIDON_GINGIVA = 17
# 나머지 둘은 악궁 전체가 반대쪽이다 (치아 11/11, 14/14가 정중선 너머 최대 25 mm.  superior를
# 치은 위치로 재도 치관 normal로 재도 같다).  network는 좌우를 기하로 배우므로 label을 기하에
# 맞춰 좌우를 바꾼다.  이 목록에서 빼고 다시 import하면 데이터셋 원래 label이 된다.
POSEIDON_SWAPPED_SIDES = {("000117", "maxilla"), ("999994", "mandible")}


def poseidon_slot_fdi(slot: int, jaw_upper: bool, swap_sides: bool = False) -> int:
    """region_mapping 슬롯 1..16 -> FDI.  1..8 = 환자 왼쪽 8번..1번, 9..16 = 오른쪽 1번..8번
    (swap_sides면 반대)."""
    first_half = slot <= 8
    tooth = 9 - slot if first_half else slot - 8      # 중절치 1 .. 제3대구치 8
    left = first_half != swap_sides
    quadrant = (2 if left else 1) if jaw_upper else (3 if left else 4)
    return quadrant * 10 + tooth


class _NoGlobals(pickle.Unpickler):
    """Poseidon3D label pkl은 int list뿐이다.  어떤 함수 / 클래스도 불러오지 못하게 막는다
    (pickle은 불러오는 순간 임의 코드를 실행할 수 있다)."""

    def find_class(self, module: str, name: str) -> object:
        raise pickle.UnpicklingError(f"refusing global {module}.{name} in a label file")


def _load_pickle(path: Path) -> object:
    with open(path, "rb") as f:
        return _NoGlobals(f).load()


# ------------------------------------------------------------------- labels
def vertex_vote(face_labels: np.ndarray, faces: np.ndarray, n_vert: int) -> np.ndarray:
    """face label -> 정점 label: 정점을 둘러싼 face의 다수결, 동수면 치아 쪽 (0이 아닌 쪽).
    어느 face에도 안 속한 정점은 0 (gingiva)."""
    labels = np.unique(face_labels)                                # 오름차순, gingiva(0)가 먼저
    slot = np.searchsorted(labels, face_labels)
    votes = np.zeros((n_vert, len(labels)))
    for corner in range(3):
        np.add.at(votes, (faces[:, corner], slot), 1.0)
    if labels[0] == 0:
        votes[:, 1:] += 0.5 * (votes[:, 1:] > 0)                   # 동수면 치아 쪽 (0.5 < 표 하나)
    out = labels[votes.argmax(1)]
    out[votes.sum(1) == 0] = 0
    return out


def fdi_instances(fdi: np.ndarray) -> np.ndarray:
    """치아(FDI)마다 instance 하나: 0 gingiva, 1..K (FDI 오름차순)."""
    teeth = np.unique(fdi[fdi > 0])
    return np.where(fdi > 0, np.searchsorted(teeth, fdi) + 1, 0)


def iosseg_vertex_fdi(face_colours: np.ndarray, faces: np.ndarray, positions: np.ndarray,
                      jaw_upper: bool) -> tuple[np.ndarray, int]:
    """3D-IOSSeg face label 색 -> 정점 FDI (0 gingiva), 그리고 팔레트 밖이라 다시 매긴 face 수.

    팔레트 색의 변형은 가장 가까운 팔레트 색으로, 팔레트에서 먼 색 (아래턱에 위턱 색이
    묻은 조각 등)은 위치가 가장 가까운 정상 face의 label로 바꾼다.
    """
    palette = {c: f for c, f in IOSSEG_FDI.items() if (f // 10 in (1, 2)) == jaw_upper}
    palette[IOSSEG_GINGIVA] = 0
    keys, values = np.array(list(palette), np.int64), np.array(list(palette.values()), np.int64)
    d2 = ((face_colours[:, None, :].astype(np.int64) - keys[None]) ** 2).sum(-1)       # [F,P]
    nearest = d2.argmin(1)
    gap = d2[np.arange(len(faces)), nearest]
    face_fdi = values[nearest]
    far = gap > IOSSEG_COLOUR_TOLERANCE ** 2
    if far.any():
        centres = positions[faces].mean(1)
        _, j = cKDTree(centres[~far]).query(centres[far])
        face_fdi[far] = face_fdi[~far][j]
    return vertex_vote(face_fdi, faces, len(positions)), int((gap > 0).sum())


# -------------------------------------------------------------------- write
def write_raw(path: Path, mesh: ScanDecoderOutput, fdi: np.ndarray | None, instances: np.ndarray | None,
              source: str) -> None:
    """data/ios raw PLY 한 파일.  label 범주는 FDI에서 정한다 (지금의 오픈 데이터셋에는 치은과 치아뿐).
    색은 실제 색일 때만 남긴다 (Teeth3DS의 0.502 회색은 버린다)."""
    rgb = real_colour(mesh.colours)
    labels = None if fdi is None else np.where(fdi > 0, VertexLabel.TOOTH, VertexLabel.GINGIVA)
    inp = ScanEncoderInput(path, mesh.positions, mesh.faces,
                           None if rgb is None else torch.from_numpy(rgb.astype(np.float32) / 255.0),
                           *(None if t is None else torch.from_numpy(t.astype(np.int64)) for t in (fdi, labels, instances)),
                           ("dental_ai ios raw v2", f"source {source}"))
    if not ply_encoder_cpu(inp, ScanEncoderOutput()):
        raise ValueError("could not encode the PLY (labels out of range or length mismatch)")


def _decode(fn: Callable[[ScanDecoderInput, ScanDecoderOutput], bool], path: Path) -> ScanDecoderOutput:
    mesh = ScanDecoderOutput()
    if not fn(ScanDecoderInput(path), mesh) or mesh.faces is None:
        raise ValueError("could not decode a mesh")
    return mesh


# ---------------------------------------------------------------- importers
def import_teeth3ds(obj_path: Path, out_path: Path) -> tuple[str | None, str | None]:
    """Teeth3DS obj (+ 같은 이름 json) -> raw PLY.  json이 없으면 (3DTeethLand 테스트) label 없이."""
    try:
        mesh = _decode(obj_decoder_cpu, obj_path)
        json_path = obj_path.with_suffix(".json")
        if not json_path.exists():
            write_raw(out_path, mesh, None, None, f"Teeth3DS {obj_path.name}")
            return None, "no labels (3DTeethLand test scan)"
        d = json.loads(json_path.read_text(encoding="utf-8"))
        labels, ins = np.asarray(d["labels"], np.int64), np.asarray(d["instances"], np.int64)
        if len(labels) != mesh.positions.shape[0]:
            return f"label/vertex mismatch {len(labels)} vs {mesh.positions.shape[0]}", None
        write_raw(out_path, mesh, labels, ins, f"Teeth3DS {obj_path.name} + {json_path.name}")
        return None, None
    except Exception as e:                       # 깨진 스캔 하나가 전체를 멈추면 안 된다
        return repr(e), None


def import_3d_iosseg(ply_path: Path, out_path: Path) -> tuple[str | None, str | None]:
    """3D-IOSSeg ply (label이 face 색) -> raw PLY."""
    try:
        jaw_upper = IOSSEG_NAME.fullmatch(ply_path.name).group(2) == "U"
        mesh = _decode(ply_decoder_cpu, ply_path)
        if mesh.face_colours is None:
            return "no face label colours", None
        fdi, remapped = iosseg_vertex_fdi(mesh.face_colours.numpy(), mesh.faces.numpy(), mesh.positions.numpy(),
                                          jaw_upper)
        write_raw(out_path, mesh, fdi, fdi_instances(fdi), f"3D-IOSSeg {ply_path.parent.name}/{ply_path.name}")
        return None, (f"{remapped} faces had an off-palette label colour" if remapped else None)
    except Exception as e:
        return repr(e), None


def import_poseidon3d(stl_path: Path, regions_path: Path, mapping: list[int], jaw_upper: bool, swap_sides: bool,
                      out_path: Path) -> tuple[str | None, str | None]:
    """Poseidon3D stl + face 영역 pkl + 영역 -> 슬롯 표 -> raw PLY."""
    try:
        slot_of = {r: s for s, r in enumerate(mapping) if r >= 0}
        if not slot_of:
            return "skipped: region_mapping is all -1, so the teeth cannot be numbered", None
        mesh = _decode(stl_decoder_cpu, stl_path)
        regions = np.asarray(_load_pickle(regions_path), np.int64)
        if len(regions) != mesh.faces.shape[0]:
            return f"region/face mismatch {len(regions)} vs {mesh.faces.shape[0]}", None
        table = {POSEIDON_GINGIVA: 0, **{r: poseidon_slot_fdi(s, jaw_upper, swap_sides) for r, s in slot_of.items()}}
        unknown = ~np.isin(regions, list(table))
        if unknown.any():
            return f"{int(unknown.sum())} faces have a region with no slot ({np.unique(regions[unknown]).tolist()})", None
        face_fdi = np.vectorize(table.get, otypes=[np.int64])(regions)
        fdi = vertex_vote(face_fdi, mesh.faces.numpy(), mesh.positions.shape[0])
        write_raw(out_path, mesh, fdi, fdi_instances(fdi), f"Poseidon3D {stl_path.name} + {regions_path.name}"
                  + (" (left/right swapped to match the geometry)" if swap_sides else ""))
        return None, ("left/right swapped to match the geometry" if swap_sides else None)
    except Exception as e:
        return repr(e), None


def import_cbct_ios_fusion(stl_path: Path, labels_path: Path, out_path: Path) -> tuple[str | None, str | None]:
    """CBCT-IOS 융합 (Liu 2023) IOSWithGT stl + face FDI txt -> raw PLY."""
    try:
        mesh = _decode(stl_decoder_cpu, stl_path)
        face_fdi = np.loadtxt(labels_path, dtype=np.int64, ndmin=1)
        if len(face_fdi) != mesh.faces.shape[0]:
            return f"label/face mismatch {len(face_fdi)} vs {mesh.faces.shape[0]}", None
        fdi = vertex_vote(face_fdi, mesh.faces.numpy(), mesh.positions.shape[0])
        write_raw(out_path, mesh, fdi, fdi_instances(fdi), f"CBCT-IOS fusion {stl_path.parent.name}/{stl_path.name}")
        return None, None
    except Exception as e:
        return repr(e), None


# --------------------------------------------------------------------- jobs
Job = tuple[Callable[..., tuple[str | None, str | None]], tuple]


def teeth3ds_jobs(src: Path, raw: Path) -> tuple[list[Job], dict[str, list[str]]]:
    objs = sorted(src.glob("data_part_*/*/*/*.obj"))
    return [(import_teeth3ds, (o, raw / f"{o.stem}.ply")) for o in objs], {}    # split: teeth3ds/splits 그대로


def iosseg_jobs(src: Path, raw: Path) -> tuple[list[Job], dict[str, list[str]]]:
    plys = sorted(p for p in src.glob("*/*.ply") if IOSSEG_NAME.fullmatch(p.name))
    jobs, splits = [], {}
    for p in plys:
        m = IOSSEG_NAME.fullmatch(p.name)
        name = f"IOSSEG{m.group(1)}_{'upper' if m.group(2) == 'U' else 'lower'}"
        jobs.append((import_3d_iosseg, (p, raw / f"{name}.ply")))
        splits.setdefault(p.parent.name, []).append(name)        # train / test 폴더가 split이다
    return jobs, splits


def poseidon_jobs(src: Path, raw: Path) -> tuple[list[Job], dict[str, list[str]]]:
    jobs, names = [], {}
    for m in json.loads((src / "metadata.json").read_text()):
        for jaw, upper in (("maxilla", True), ("mandible", False)):
            if not m[f"{jaw}_paths"]:
                continue
            name = f"POSEIDON{m['id']}_{'upper' if upper else 'lower'}"
            stl = src / m["id"] / Path(m[f"{jaw}_paths"][0]).name
            jobs.append((import_poseidon3d, (stl, src / m["id"] / f"{m['id']}_REGIONS_FACES_{jaw}.pkl",
                                             m[f"{jaw}_region_mapping"], upper,
                                             (m["id"], jaw) in POSEIDON_SWAPPED_SIDES, raw / f"{name}.ply")))
            names.setdefault(m["id"], []).append(name)
    splits = {}
    for line in (src / "split.txt").read_text().split():
        split, ids = line.split(":", 1)
        splits[split] = [n for i in ids.split(",") for n in names.get(i, [])]
    return jobs, splits


def fusion_jobs(src: Path, raw: Path) -> tuple[list[Job], dict[str, list[str]]]:
    jobs = []
    for stl in sorted(src.glob("*/[lu]_aligned.stl")):
        jaw = "lower" if stl.name.startswith("l") else "upper"
        jobs.append((import_cbct_ios_fusion, (stl, stl.parent / f"{stl.name[0]}.txt",
                                              raw / f"FUSION{stl.parent.name}_{jaw}.ply")))
    return jobs, {}                              # 공식 split 없음


JOBS = {"teeth3ds": teeth3ds_jobs, "3d_iosseg": iosseg_jobs, "poseidon3d": poseidon_jobs,
        "cbct_ios_fusion": fusion_jobs}


def _run(job: Job) -> tuple[Path, str | None, str | None]:
    fn, args = job
    return (args[-1], *fn(*args))


def _worker_init() -> None:
    torch.set_num_threads(1)                     # worker가 여럿이니 worker마다 1 thread


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", choices=DATASETS)
    ap.add_argument("--src", type=Path, required=True, help="원본 폴더")
    ap.add_argument("--out", type=Path, required=True, help="data/ios/<데이터셋> (raw/, splits/를 만든다)")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0, help="앞의 N개 스캔만 (smoke test)")
    ap.add_argument("--force", action="store_true", help="이미 있는 raw PLY도 다시 만든다")
    a = ap.parse_args()
    raw = a.out / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    jobs, splits = JOBS[a.dataset](a.src, raw)
    if a.limit:
        jobs = jobs[:a.limit]
    todo = [j for j in jobs if a.force or not j[1][-1].exists()]
    print(f"{len(jobs)} scans, {len(todo)} to import -> {raw}", flush=True)
    t0, errs, notes = time.time(), [], []
    with Pool(a.workers, initializer=_worker_init) as pool:
        for k, (dst, err, note) in enumerate(pool.imap_unordered(_run, todo, chunksize=2), 1):
            if err:
                errs.append((dst, err))
            if note:
                notes.append((dst, note))
            if k % 100 == 0 or k == len(todo):
                el = time.time() - t0
                print(f"[{k}/{len(todo)}] {el / 60:.1f} min  eta {(len(todo) - k) * el / k / 60:.1f} min  "
                      f"errs {len(errs)}", flush=True)
    print(f"done in {(time.time() - t0) / 60:.1f} min, {len(errs)} errors / skips")
    for d, e in errs:
        print("  error", d.name, e)
    for note, count in sorted({n: sum(1 for _, x in notes if x == n) for _, n in notes}.items()):
        print(f"  note  {count} scans: {note}")

    if splits:
        (a.out / "splits").mkdir(exist_ok=True)
        for split, names in splits.items():
            names = [n for n in names if (raw / f"{n}.ply").exists()]
            (a.out / "splits" / f"{split}.txt").write_text("\n".join(names) + "\n")
            print(f"splits/{split}.txt: {len(names)} scans")


if __name__ == "__main__":
    main()
