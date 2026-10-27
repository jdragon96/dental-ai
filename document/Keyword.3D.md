# 3D Segmentation Keyword

3D 분할(특히 치아 instance segmentation) 문서·코드·논문에서 자주 나오는 단어를 모았다.
표의 **이 프로젝트에서** 열은 그 개념이 `toothseg3d` / `segmentation3d` 어디에 나오는지다.

## 0. 파이프라인 위에서 보기

```
구강 스캔 (mesh)
   │ scan_decoder          vertex / face / colour 읽기
   │ voxel_downsample      voxelization, voxel size (250 µm), inverse index
   │ normal_estimation     normal
   │ model_input           serialization (Morton curve), patch, pooling 계층
   ▼
백본 (PTv3 계열 U-Net)      encoder / decoder, skip connection, patch attention, CPE
   ▼
instance_network           query, masked cross-attention, class head / mask head
   │ query_fusion          TTA, Hungarian matching (패스 간 쿼리 맞추기)
   │ mask_upsample         voxel copy / soft k-NN 보간
   │ instance_extraction   score threshold, mask score, erode, min-frac
   ▼
result_encoder             정점별 instance id / label
```

---

## 1. 데이터 표현

| 용어 | 뜻 | 이 프로젝트에서 |
| --- | --- | --- |
| Point Cloud (포인트 클라우드) | (x, y, z) 점들의 집합. 순서도, 연결 정보도 없다 | 네트워크 입력은 포인트 클라우드로 다룬다 |
| Mesh (메시) | Vertex(정점) + Face(삼각형 면) + Edge(모서리). 표면의 연결 정보가 있다 | 구강 스캔 원본 (OBJ / PLY / STL) |
| Vertex label / Face label | 라벨을 정점에 붙이느냐, 면에 붙이느냐 | Teeth3DS는 정점 라벨, MeshSegNet은 face 기반 |
| Voxel (복셀) | 3D 격자의 한 칸. 2D 픽셀의 3D 버전 | |
| Volume (볼륨) | CT / CBCT처럼 격자 전체에 값이 들어 있는 3D 영상 | 구강 스캔(표면)과는 다른 도메인. 3D U-Net / nnU-Net 계열이 쓰인다 |
| Normal (법선) | 표면에 수직인 단위 벡터 | 입력 특징 = xyz + normal + jaw flag |
| Feature / Channel | 점마다 붙은 벡터. 입력 특징(xyz, normal, colour) → 네트워크가 만든 고차원 특징 | `[N, 64]` 포인트 특징 |
| Intraoral Scan (IOS, 구강 스캔) | 구강 스캐너로 얻은 치아·치은 표면 메시 | 사내 L-SCAN |
| Jaw / Arch (악궁) | Upper(상악, maxilla) / Lower(하악, mandible) | jaw flag |
| Gingiva (치은) | 잇몸. 보통 label 0 = 배경 | |
| FDI notation | 두 자리 치아 번호 = 사분면(1–4) + 위치(1–8). 예: 11 = 우상악 중절치, 38 = 좌하악 사랑니 | 17클래스 모델의 출력 |
| Canonical frame (정준 좌표계) | 데이터셋이 맞춰 둔 기준 자세 (교합면 방향, 전치 방향 등) | Teeth3DS는 정렬돼 있고 사내 스캔은 아니다 |

## 2. Task 종류

| 용어 | 뜻 | 이 프로젝트에서 |
| --- | --- | --- |
| Semantic Segmentation | 점마다 클래스 하나. 같은 클래스의 서로 다른 개체는 구분하지 않는다 | `toothseg3d_300um.pt` (FDI 번호 = 클래스) |
| Instance Segmentation | 개체마다 따로 마스크. "치아 1, 치아 2, …"를 구분한다 | `toothinst_250um.pt` |
| Panoptic Segmentation | semantic + instance. 셀 수 없는 것(stuff, 치은)은 클래스만, 셀 수 있는 것(thing, 치아)은 개체까지 | |
| Part Segmentation | 한 물체를 부위로 나눈다 (ShapeNet-Part) | |
| Tooth Identification / Labeling | 분할된 치아에 FDI 번호를 붙인다 | 인스턴스 모델은 하지 않는다 |
| Tooth Localization | 치아 중심 위치를 찾는다 | TLA 지표 |
| Foreground / Background | 관심 대상(치아) / 나머지(치은) | |

## 3. 샘플링 · 전처리

| 용어 | 뜻 | 이 프로젝트에서 |
| --- | --- | --- |
| Voxelization / Voxel downsampling (Grid sampling) | 공간을 격자로 나누고 칸마다 점 하나만 남긴다 | `voxel_downsample` |
| Voxel size / Resolution | 격자 한 칸의 크기. **작을수록 고해상도**, 점이 많고 느리다 | 250 µm / 300 µm |
| Inverse index | 원본 정점 → 그 정점이 속한 복셀 번호. 업샘플할 때 쓴다 | `MaskUpsampleInput.inverse` |
| FPS (Farthest Point Sampling) | 이미 고른 점들에서 가장 먼 점을 반복해서 골라, 고르게 퍼진 부분집합을 얻는다 | 쿼리 초기화 위치 |
| Random sampling | 무작위 추출. 빠르지만 밀도가 높은 곳에 몰린다 | |
| k-NN (k-Nearest Neighbors) | 가장 가까운 k개 이웃 | soft k-NN 업샘플 (k=5) |
| Ball query / Radius search | 반경 r 안의 이웃. **여기서 query는 "질의점"이라는 뜻** — §5의 Query와 다르다 | PointNet++ |
| KD-tree | k-NN / 반경 탐색용 공간 자료구조 | `cKDTree` |
| Normalization | 중심 이동(centering) + 스케일 정규화 | xyz / 35 mm |
| Normal estimation | 주변 점이나 면으로 법선을 계산 | `normal_estimation` |
| Curvature (곡률) | 표면이 휘는 정도. 치아–치은 경계에서 크게 바뀐다 | |
| Oracle upper bound | 정답 라벨을 다운샘플 → 원본 복원만 해도 얻는 최고 점수. 그 해상도에서 어떤 모델도 못 넘는 천장 | `resolution_study.py` |

## 4. 백본 (Backbone)

**Backbone**: 입력 점에서 점별 특징을 뽑는 몸통 네트워크.
그 위에 붙어 최종 출력을 내는 작은 모듈은 **Head**라고 한다 (semantic head, offset head, instance head).

| 용어 | 뜻 | 이 프로젝트에서 |
| --- | --- | --- |
| PointNet | 점마다 MLP → max pooling. 점 순서에 무관한(permutation invariant) 네트워크의 시작 | |
| PointNet++ | Set Abstraction(FPS + ball query + PointNet)으로 줄이고, Feature Propagation(역거리 가중 보간)으로 키운다 | ToothGroupNetwork의 백본 |
| DGCNN / EdgeConv | 층마다 특징 공간에서 k-NN 그래프를 다시 만들고(dynamic graph) 엣지 특징을 모은다 | |
| Sparse Convolution | 비어 있지 않은 복셀에서만 컨볼루션 (MinkowskiEngine, spconv). Submanifold sparse conv는 출력 위치를 입력과 같게 유지한다 | 의존성 때문에 미사용 |
| Point Transformer (V1 / V2 / V3) | 점 이웃 간 어텐션. V3는 이웃 탐색 대신 직렬화 + 패치 어텐션으로 단순·고속화했다 | 백본 (PTv3 계열) |
| Serialization (직렬화) | 3D 점을 space-filling curve 순서로 1차원 정렬 → 공간에서 가까운 점이 배열에서도 가깝다 | `model_input` |
| Space-filling curve | 공간 전체를 한 줄로 훑는 곡선. Z-order(Morton), Hilbert | |
| Morton code | 정수 좌표 x, y, z의 비트를 번갈아 끼워 만든 정렬 키 = Z-order | `morton3d_curve_cpu/gpu` |
| Patch attention | 직렬화된 배열을 고정 길이로 끊어 패치 안에서만 어텐션 | patch = 1024 |
| Shuffle order | 블록마다 다른 곡선(축 순열·반사)으로 정렬해 패치 경계를 바꾼다 → 패치 간 정보 교환 | 4종 곡선 |
| Positional Encoding | 어텐션은 위치를 모르므로 좌표 정보를 특징에 넣어 준다 | `kv_pos` |
| CPE (Conditional Positional Encoding) | 이웃 특징으로 위치 정보를 만들어 더한다. PTv3는 sparse conv로 만든다(xCPE) | depthwise Conv1d + 상대좌표 MLP로 대체 |
| U-Net (Encoder–Decoder) | 해상도를 줄이며 문맥을 모으고(encoder), 다시 키우며 세부를 복원(decoder) | enc0–enc4 / dec0–dec3 |
| Skip connection | 같은 해상도의 encoder 특징을 decoder에 이어 붙인다 | |
| Grid pooling / Unpooling | 복셀을 2배로 키워 평균 / 부모 인덱스로 되돌림 | |
| Receptive field | 한 출력이 볼 수 있는 입력 범위 | 최하단(4.8 mm)은 전역 어텐션 |
| Packed layout | 점 수가 다른 스캔들을 패딩 없이 한 축에 이어 붙이고, 경계를 인덱스로 관리 | 배치 처리 방식 |

---

## 5. 쿼리 기반 Instance Segmentation

### 5.1. Query란

**Query** = 학습되는 벡터 K개 (여기서는 32개). 각 쿼리는 **"물체 하나를 찾아 맡는 슬롯"** 이다.
쿼리는 디코더 층을 거치며 cross-attention으로 포인트 특징을 읽고, 끝나면 쿼리마다 두 가지를 낸다.

- **class**: 이 쿼리가 치아를 찾았는가 / 아무것도 아닌가 (no-object)
- **mask**: 어떤 점들이 그 치아인가 (점 임베딩 · 쿼리 임베딩)

치아 개수는 스캔마다 다르지만, 쿼리를 넉넉히 두고 남는 쿼리는 no-object로 버리면 된다.
클러스터링이나 NMS 같은 후처리가 거의 필요 없다.

```
point features [N, D] ──────────────────────┐ (Key, Value)
                                             ▼
queries [K, D] ── masked cross-attn → self-attn → FFN   (× L층)
                                             │
                 ├─ class head → [K, 2]      (치아 / ∅)
                 └─ mask = point_emb · query_emb / √d → [N, K]
```

### 5.2. Attention의 Q / K / V

`attention = softmax(Q·Kᵀ / √d) · V`

| 용어 | 뜻 | 쿼리 디코더의 cross-attention에서 |
| --- | --- | --- |
| Query (Q) | "무엇을 찾나" — 묻는 쪽 | 인스턴스 쿼리 |
| Key (K) | "나는 이런 것" — 비교 대상 | 포인트 특징 |
| Value (V) | 실제로 가져갈 내용 | 포인트 특징 |

### 5.3. 용어

| 용어 | 뜻 | 이 프로젝트에서 |
| --- | --- | --- |
| Object query / Instance query | 5.1의 쿼리 | `n_query = 32` |
| Learnable query vs Non-parametric query | `nn.Embedding`으로 학습되는 쿼리 vs 실제 점(FPS)의 특징·위치로 만든 쿼리 (Mask3D) | FPS 포인트 특징으로 초기화 |
| Self-attention | 쿼리끼리 어텐션 → 두 쿼리가 같은 치아를 잡지 않도록 조율 | |
| Cross-attention | 쿼리(Q)가 포인트 특징(K, V)을 읽는다 | |
| Masked cross-attention | 이전 층이 예측한 마스크 바깥은 못 보게 막고 어텐션 (Mask2Former) | `attn_bias` |
| Attention mask / bias | 어텐션 점수에 더하는 값. `-inf`면 그 위치를 무시 | TensorRT fp16에서 NaN → `-1e4`로 교체 |
| Multi-head attention | 채널을 여러 head로 나눠 병렬 어텐션 | heads = 8 |
| FFN (Feed-Forward Network) | 어텐션 뒤에 붙는 2층 MLP | |
| Pre-norm / Residual | LayerNorm을 블록 앞에 두고, 입력을 출력에 더한다 | |
| Decoder layer | cross-attn → self-attn → FFN 한 묶음 | layers = 4 |
| Mask head / Mask embedding | 점 임베딩과 쿼리 임베딩의 내적으로 마스크 로짓을 만든다 | `/ √d` 스케일 + 학습 가능 bias |
| No-object (∅) class | "이 쿼리는 아무것도 못 찾았다" 클래스 | |
| Set prediction | 출력을 순서 없는 집합으로 보고 정답 집합과 1:1로 맞춰 학습한다 (DETR) | |
| Hungarian matching (Bipartite matching) | 예측 K개 ↔ 정답 M개를 비용 합이 최소가 되게 1:1 배정 | 학습 손실, TTA 쿼리 정렬 |
| Matching cost | 매칭에 쓰는 비용 = class + BCE + dice 가중합 | |
| Deep supervision (Auxiliary loss) | 마지막 층뿐 아니라 각 디코더 층 출력에도 손실을 건다 | 층별 거친 마스크 감독 |
| Mask score | 쿼리 최종 점수 = P(치아) × 마스크 내부 평균 확률. "찾았나"에 "마스크가 좋은가"를 더한다 | `mask_confidence` (+1.0 mAP) |
| NMS (Non-Maximum Suppression) | 겹치는 예측 중 점수 높은 것만 남긴다. 쿼리 방식은 매칭 학습 덕에 보통 필요 없다 | |

### 5.4. 다른 Instance Segmentation 계열

| 계열 | 방식 | 예 |
| --- | --- | --- |
| Proposal-based (top-down) | 박스·영역 후보를 먼저 내고, 후보마다 안쪽을 분할 | 3D-BoNet, GSPN |
| Grouping-based (bottom-up) | 점마다 자기 물체 중심으로의 offset을 예측해 모은 뒤 클러스터링 | PointGroup, ToothGroupNetwork |
| Query-based (transformer) | 쿼리가 마스크를 직접 예측. 후처리 최소 | Mask3D, SPFormer, OneFormer3D, **toothinst** |

| 용어 | 뜻 | 이 프로젝트에서 |
| --- | --- | --- |
| Centroid offset / Center voting | 점 → 자기 치아 중심까지의 벡터를 회귀 | offset head (보조 과제) |
| Clustering | offset으로 모인 점들을 덩어리로 묶는다 (DBSCAN, BFS 등) | 17클래스 모델은 필요 없음 |
| Superpoint | 기하적으로 비슷한 점을 미리 묶은 덩어리. 그 단위로 예측한다 (SPFormer) | |

---

## 6. 손실 함수

| 용어 | 뜻 | 이 프로젝트에서 |
| --- | --- | --- |
| Logit | sigmoid / softmax 전의 값 | |
| Cross-Entropy (CE) | 다중 클래스 분류 손실 | semantic head, 쿼리 class |
| BCE (Binary Cross-Entropy) | 점마다 "이 마스크에 속하나"를 이진 분류 | 마스크 손실 (가중치 5.0) |
| Dice loss | 1 − Dice. 영역 겹침을 직접 최적화하고 클래스 불균형에 강하다 | 가중치 2.0 |
| Focal loss | 쉬운 샘플의 가중치를 낮춰 어려운 샘플에 집중 | |
| Class weight | 드문 클래스(사랑니 등)에 가중치를 더 준다 | inverse-sqrt 빈도 |
| Offset loss | 예측 offset과 정답 벡터의 L1 (+ 방향 cosine) | |

## 7. 학습 · 증강

| 용어 | 뜻 | 이 프로젝트에서 |
| --- | --- | --- |
| Augmentation (증강) | 학습할 때 입력을 변형: 회전, 스케일, jitter(작은 노이즈), crop, color jitter | |
| SO(3) | 3D 회전 전체의 집합. "SO(3) 증강" = 임의 방향 회전 | 정렬 없이 쓰는 인스턴스 모델의 핵심 |
| Invariance (불변) | 입력을 돌려도 출력이 같다 (라벨, 마스크) | |
| Equivariance (등변) | 입력을 돌리면 출력도 똑같이 돈다 (offset 벡터, 법선) | |
| Mirror / Flip | 좌우 반전. FDI 번호를 쓸 때는 좌우 번호도 바꿔야 한다 | |
| Crop augmentation | 일부만 잘라 부분 스캔을 흉내 | 치아 단위 crop |
| Pretraining / Fine-tuning | 큰 데이터로 학습 → 목표 데이터로 추가 학습 | Teeth3DS → 사내 라벨 |
| Self-supervised pretraining | 라벨 없이 사전학습 (Sonata, MSC 등) | 보류 |
| Transfer / Zero-shot | 다른 도메인에 적용 / 추가 학습 없이 그대로 적용 | 사내 스캔 전이 |
| Domain gap | 학습 데이터와 실제 데이터의 차이 (스캐너, 색상, 라벨 관행) | |
| Overfit test | 고정 배치 하나에 과적합되는지 보고 손실·파이프라인을 검증 | 매칭 IoU 0.046 → 0.988 |
| Cold start | 초기화가 나빠 학습 초반을 허비 | 백본 콜드 스타트 (mAP 2.45) |
| Hold-out / Split | 학습에 쓰지 않은 평가용 데이터 | Teeth3DS test 600 |
| Model soup (Weight averaging) | 같은 구조의 두 체크포인트 가중치를 선형 보간 | `soup.py` |

## 8. 추론 · 후처리

| 용어 | 뜻 | 이 프로젝트에서 |
| --- | --- | --- |
| TTA (Test-Time Augmentation) | 입력을 여러 번 변형(회전)해 추론하고 결과를 평균 | ×8 |
| Query fusion | TTA 패스마다 쿼리 순서가 다르므로 마스크 겹침으로 헝가리안 매칭한 뒤 평균 | `query_fusion` |
| Upsampling / Interpolation | 다운샘플 해상도의 예측을 원본 정점으로 되돌림 | `mask_upsample` |
| Voxel copy upsample | 복셀의 라벨을 그 복셀의 모든 정점에 복사. 빠르지만 경계가 계단처럼 된다 | `--upsample voxel` |
| Soft k-NN upsample | 소프트 확률을 k-NN 거리 가중으로 보간한 뒤 결정 | +1.5 mAP |
| Soft mask / Hard mask | 확률값 마스크 / 0·1 마스크 | |
| Argmax / Threshold | 가장 큰 클래스를 고름 / 확률이 임계값을 넘는지 | `score_thresh = 0.5` |
| Connected component (연결 성분) | 서로 이어진 점 덩어리. 가장 큰 것만 남겨 떨어진 오탐을 지운다 | 17클래스 모델 후처리 |
| Erode / Dilate | 마스크 경계를 깎음 / 키움 | `erode_mm` (사내 라벨 관행 맞추기) |
| Min-frac filter | 전체 대비 너무 작은 인스턴스 제거 | `min_frac = 0.005` |
| Graph smoothing / CRF / Graph cut | 이웃 그래프 위에서 라벨을 매끈하게 정리. 법선이 꺾이는 곳에서 끊으면 경계가 기하 엣지에 붙는다 | `mesh_smooth.py` |

---

## 9. 평가 지표

| 용어 | 뜻 | 비고 |
| --- | --- | --- |
| IoU (Intersection over Union) | 교집합 / 합집합 | |
| mIoU | 클래스별 IoU 평균 | semantic 지표 |
| Dice / DSC | 2 × 교집합 / (예측 크기 + 정답 크기). F1과 같은 식 | |
| Precision / Recall / F1 | 예측 중 맞은 비율 / 정답 중 찾은 비율 / 둘의 조화평균 | |
| AP (Average Precision) | 예측을 점수 순으로 정렬해 그린 PR 곡선 아래 면적. 정답과 IoU가 임계값을 넘으면 TP | 그래서 점수(mask score)가 중요하다 |
| AP50 / AP25 | IoU 0.5 / 0.25 기준 AP. "찾았나"에 가깝다 | |
| mAP@[.50:.95] | IoU 0.50~0.95(0.05 간격)의 AP 평균. 경계 정확도까지 반영 | 인스턴스 모델 주 지표 |
| TSA (Teeth Segmentation Accuracy) | 치아 / 치은 이진 분할의 F1 | 3DTeethSeg'22 |
| TLA (Teeth Localization Accuracy) | 예측·정답 치아 중심 거리를 치아 크기로 정규화한 평균. 낮을수록 좋고 점수엔 exp(−TLA)로 들어간다 | 3DTeethSeg'22 |
| TIR (Teeth Identification Rate) | 위치가 맞고(중심 거리 < 치아 크기 절반) FDI 번호까지 맞춘 치아의 비율 | 3DTeethSeg'22 |
| Challenge score | (exp(−TLA) + TSA + TIR) / 3 | 3DTeethSeg'22 |

## 10. 배포

| 용어 | 뜻 | 이 프로젝트에서 |
| --- | --- | --- |
| ONNX | 프레임워크 중립 모델 포맷 | `export_onnx.py` |
| TensorRT / Engine | NVIDIA 추론 최적화기 / 그 결과물 (특정 GPU용 바이너리) | |
| fp32 / fp16 / bf16 / int8 | 수치 정밀도. 낮을수록 빠르고 작지만 오차·overflow 위험 | |
| Quantization (양자화) | 가중치·활성값을 int8 등 낮은 정밀도로 표현 | |
| QDQ | Quantize–DeQuantize 노드를 그래프에 명시하는 양자화 방식 | `quantize_qdq.py` |
| Calibration | 대표 데이터로 int8 변환 범위(scale)를 정함 | `int8_calib.py` |
| Dynamic shape | 점 수가 스캔마다 달라 min / opt / max 범위로 엔진을 빌드 | `trt_shapes.py` |
| Latency / Throughput | 1건 처리 시간 / 단위 시간당 처리량 | |

---

## 11. 데이터셋 · 대표 모델

### 11.1. 데이터셋

| 이름 | 내용 |
| --- | --- |
| Teeth3DS / 3DTeethSeg'22 | 구강 스캔 1800개 (900명 × 상·하악), FDI 정점 라벨. MICCAI 2022 챌린지 |
| ScanNet / ScanNet200 | 실내 RGB-D 재구성 장면. 3D instance segmentation 표준 벤치마크 |
| S3DIS | 실내 공간 포인트 클라우드 |
| ShapeNet-Part | 물체 부위 분할 |

### 11.2. 모델

| 이름 | 연도 | 한 줄 |
| --- | --- | --- |
| PointNet / PointNet++ | 2017 | 점을 직접 다루는 네트워크의 시초 |
| DGCNN | 2019 | 동적 k-NN 그래프 + EdgeConv |
| MinkowskiNet | 2019 | sparse convolution |
| MeshSegNet | 2020 | 치아 메시 face 분할. 사내 기존 모델 |
| PointGroup | 2020 | offset + 클러스터링 인스턴스 분할 |
| DETR | 2020 | (2D) 쿼리 + 헝가리안 매칭의 시초 |
| TSegNet | 2021 | 치아 중심을 먼저 찾고 치아별로 분할 |
| Mask2Former | 2022 | (2D) masked cross-attention |
| ToothGroupNetwork | 2022 | 3DTeethSeg'22 1위. semantic + centroid offset |
| Mask3D | 2023 | 3D 쿼리 기반 인스턴스 분할. FPS 쿼리 |
| SPFormer | 2023 | superpoint + 쿼리 디코더 |
| Point Transformer V3 (PTv3) | 2024 | 직렬화 + 패치 어텐션. 이 프로젝트 백본의 원형 |
| OneFormer3D | 2024 | semantic / instance / panoptic 통합 쿼리 모델 |
| Sonata | 2025 | PTv3 자기지도 사전학습 |

---

## 12. 헷갈리기 쉬운 것

- **Query가 세 가지 뜻으로 쓰인다.**
  1. Instance query — 물체 하나를 맡는 슬롯 (§5.1)
  2. Attention의 Q — Query / Key / Value의 Q. 1의 쿼리가 cross-attention에서 Q 역할을 한다
  3. 이웃 탐색의 질의점 — ball query, k-NN query. `MaskUpsampleInput.query`(원본 정점 좌표)는 이 뜻이다
- **Decoder도 두 가지다.** U-Net decoder(해상도를 다시 키우는 부분)와 Transformer decoder(쿼리 디코더)는 다른 것이다.
- **Mask도 두 가지다.** Attention mask(못 보게 막기)와 segmentation mask(점이 이 물체에 속하는가).
- **Embedding**: `nn.Embedding`처럼 학습되는 lookup 벡터를 가리키기도 하고, 네트워크가 만든 특징 벡터 일반을 가리키기도 한다.
- **Semantic label vs Instance id**: 17클래스 모델은 클래스 = 치아라 둘이 같다. 인스턴스 모델의 id는 번호일 뿐 의미(FDI)가 없다.
- **Invariant vs Equivariant**: 라벨은 회전에 불변이어야 하고, offset 벡터는 등변이어야 한다.
- **Resolution 숫자**: voxel size가 작을수록 고해상도다 (250 µm가 300 µm보다 조밀하다).
