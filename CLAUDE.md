# Tooth Seg — 작업 규칙

## 새 코드의 위치

### Loss → `./loss/`
- loss 함수 하나당 파일 하나. 파일 이름은 loss 이름의 snake_case로 한다 (예: `dice_loss.py`).
- 같은 파일 안에 loss 양상을 볼 수 있는 시각화 코드를 함께 작성한다.
  `python loss/<name>.py`로 실행하면 그림을 같은 이름의 `.png`로 `./loss/`에 저장한다.
- 기준 예시: `loss/kl_divergence.py` (함수형 loss + `nn.Module` 래퍼, 시각화 함수, `__main__`).

### Layer → `./layer/`
- 새로운 개념의 레이어(블록·모듈)는 `./layer/`에 추가한다.

### Model → `./model/`
- 여러 오픈소스를 조합해서 만든 모델은 `./model/`에 추가한다.
