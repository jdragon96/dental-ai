# Python Coding Convention

## 0. Naming Rules (C++ → Python)

| Item | C++ | Python |
| --- | --- | --- |
| File / module | `TermCosineMatchCPU.h/.cpp` | `term_cosine_match_cpu.py` |
| Algorithm function | `TermCosineMatchCPU::Execute()` | `term_cosine_match_cpu()` (same name as its file) |
| Input / output struct | `RuleMatcherInput`, `RuleMatcherOutput` | `RuleMatcherInput`, `RuleMatcherOutput` (`@dataclass`) |
| Class | `RuleMatcher` | `RuleMatcher` (PascalCase) |
| Enum | `EColourMode` | `ColourMode` (no `E` prefix) |
| Method / function | `ExecuteBatch()` | `execute_batch()` |
| Member variable | `m_hashCache`, `m_bIsRunning` | `self._cache`, `self._stop_event` |
| Private | `private:` | leading `_` |
| CUDA kernel | `Kernel_AlgorithmA` | `kernel_algorithm_a` |
| Kernel launcher | `LaunchAlgorithmA` | `launch_algorithm_a` |
- Do not use Hungarian prefixes (`m_`, `b`, `i`, `up`, `kp`). Use type hints instead.
- Every public function and method must have type hints.
- Put `from __future__ import annotations` at the top of every module.

# 1. Algorithm Pattern

## 1.1. Folder Structure

- One algorithm = one folder.
    - `<algorithm>.py` holds only the algorithm's input / output structs.
    - Each implementation is its own file next to it.
    - Helpers shared by several implementations go in `common/`.
- Consider both CPU and GPU implementations.
    - Use the `_cpu.py` and `_gpu.py` suffixes to distinguish between them.
- Every folder is a package (`__init__.py`).
- GPU modules must be **imported lazily**, so the project still imports and runs on machines without CUDA.
    - Never import a `_gpu.py` module from `__init__.py`, from `<algorithm>.py`, or from a `_cpu.py` module.
    - Import it only at the point where the GPU path is chosen (see 1.2).

```
project/
├── interface/
│   ├── __init__.py
│   └── workflow.py               # Workflow (thread base class)
├── communication/
│   ├── __init__.py
│   ├── buffers.py                # BlockingRingBuffer, OverwriteRingBuffer, Channel
│   └── communication_module.py   # CommunicationModule
├── algorithm/
│   ├── __init__.py
│   ├── rule_matcher/
│   │   ├── __init__.py
│   │   ├── rule_matcher.py              # RuleMatcherInput, RuleMatcherOutput
│   │   ├── common/
│   │   │   ├── __init__.py
│   │   │   └── term_vector.py
│   │   ├── term_cosine_match_cpu.py     # term_cosine_match_cpu(inp, out) -> bool
│   │   └── term_cosine_match_gpu.py     # term_cosine_match_gpu(inp, out) -> bool
│   └── file_decoder/
│       ├── __init__.py
│       ├── file_decoder.py              # FileDecoderInput, FileDecoderOutput
│       ├── common/
│       │   ├── __init__.py
│       │   └── pdf_syntax.py
│       └── pdf_info_decoder_cpu.py      # pdf_info_decoder_cpu(inp, out) -> bool
└── workflow/
    ├── __init__.py
    ├── workflow1.py
    └── workflow2.py
```

## 1.2. Input / Output Struct Pattern

Every algorithm is called the same way: `fn(inp, out) -> bool`. There is no base class, strategy enum, factory or `set_strategy()`.

- `<algorithm>.py` defines `<Algorithm>Input` and `<Algorithm>Output` as `@dataclass`.
    - Every implementation of the algorithm (CPU, GPU, other methods) takes these same two structs.
- An implementation is a plain function with the signature `fn(inp: XInput, out: XOutput) -> bool`.
    - Name the function after its file: `balm_cpu.py` → `balm_cpu()`.
- The function fills `out` in place and must not modify `inp`.
- Return `True` on success.
    - Return `False` when the input cannot be processed (empty, too few points, ...). The caller must not read `out` after `False`.
    - Raise exceptions only for programming errors (wrong shape or dtype, missing field).
- Give every field of `<Algorithm>Output` a default, so `XOutput()` can be created empty.
- The caller imports the implementation it needs and calls it directly.
    - To choose CPU / GPU at runtime, use a plain `if` at the call site, and import the GPU module inside that branch.
- If an implementation keeps state that is expensive to rebuild on every call (loaded weights, CUDA streams, preallocated buffers), make it a plain class.
    - Set the state up in `__init__` and do the work in `execute(inp, out) -> bool`.
    - It still has no base class.

### Input / output structs

```python
# algorithm/bundle_adjustment/bundle_adjustment.py
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

@dataclass
class BundleAdjustmentInput:
    points: np.ndarray                          # [N,3] float32
    max_iterations: int = 20

@dataclass
class BundleAdjustmentOutput:
    poses: np.ndarray = field(default_factory=lambda: np.empty((0, 4, 4), np.float64))
    converged: bool = False
```

### Implementations

```python
# algorithm/bundle_adjustment/balm_cpu.py
from __future__ import annotations

from .bundle_adjustment import BundleAdjustmentInput, BundleAdjustmentOutput

def balm_cpu(inp: BundleAdjustmentInput, out: BundleAdjustmentOutput) -> bool:
    if inp.points.shape[0] == 0:
        return False                            # nothing to process
    # Core logic
    out.converged = True
    return True
```

```python
# algorithm/bundle_adjustment/balm_gpu.py
from __future__ import annotations

from numba import cuda                          # loaded only when this module is imported

from .bundle_adjustment import BundleAdjustmentInput, BundleAdjustmentOutput

def balm_gpu(inp: BundleAdjustmentInput, out: BundleAdjustmentOutput) -> bool:
    # host -> device, launch_xxx(...), device -> host (see 1.3)
    return True
```

### Usage

```python
from algorithm.bundle_adjustment.balm_cpu import balm_cpu
from algorithm.bundle_adjustment.bundle_adjustment import BundleAdjustmentInput, BundleAdjustmentOutput

inp = BundleAdjustmentInput(points)
out = BundleAdjustmentOutput()

if use_gpu:
    from algorithm.bundle_adjustment.balm_gpu import balm_gpu   # lazy: CUDA is needed only here
    ok = balm_gpu(inp, out)
else:
    ok = balm_cpu(inp, out)
if not ok:
    ...                                         # out must not be read
```

### Stateful implementation

```python
# algorithm/bundle_adjustment/balm_gpu.py
from __future__ import annotations

import numpy as np
from numba import cuda

from .bundle_adjustment import BundleAdjustmentInput, BundleAdjustmentOutput

class BalmGPU:
    """Keeps a CUDA stream and device buffers across calls."""

    def __init__(self, max_points: int) -> None:
        self._stream = cuda.stream()
        self._d_points = cuda.device_array((max_points, 3), np.float32, stream=self._stream)

    def execute(self, inp: BundleAdjustmentInput, out: BundleAdjustmentOutput) -> bool:
        # Core logic, reusing self._stream / self._d_points
        return True

balm = BalmGPU(max_points=100_000)              # once
ok = balm.execute(inp, out)                     # every call
```

## 1.3. CUDA Kernel Pattern

- Use **Numba** (`numba.cuda`) for kernels written in Python.
    - If an existing `.cu` kernel must be reused, use CuPy `RawKernel` with the same naming rules.
- Prefix CUDA kernel functions with `kernel_`.
- Prefix functions that launch a CUDA kernel with `launch_`.
- Kernels and launchers live in the same `_gpu.py` file, separated by section comments.
- Launchers take device arrays and a stream; host ↔ device transfer happens outside the launcher.
    - The implementation function in that file (`xxx_gpu(inp, out)`) does the transfer and calls the launcher.

```python
# algorithm/xxx/xxx_gpu.py
from __future__ import annotations

from numba import cuda

####################################################################
# Kernels
####################################################################
@cuda.jit
def kernel_algorithm_a(src, dst):
    row, col = cuda.grid(2)
    if row < src.shape[0] and col < src.shape[1]:
        # Kernel core logic
        dst[row, col] = src[row, col]

####################################################################
# Launcher
####################################################################
def launch_algorithm_a(d_src, d_dst, stream: cuda.cudadrv.driver.Stream) -> None:
    height, width = d_src.shape

    threads = (16, 16)
    blocks = (
        (height + threads[0] - 1) // threads[0],
        (width + threads[1] - 1) // threads[1],
    )

    kernel_algorithm_a[blocks, threads, stream](d_src, d_dst)
    stream.synchronize()
```

```python
# Usage
import numpy as np
from numba import cuda

stream = cuda.stream()
d_src = cuda.to_device(np.zeros((480, 640), np.float32), stream=stream)
d_dst = cuda.device_array_like(d_src, stream=stream)

launch_algorithm_a(d_src, d_dst, stream)
result = d_dst.copy_to_host(stream=stream)
```

# 2. Thread

## 2.1. Code Pattern

- Every workflow thread inherits the `Workflow` base class.
- Use `threading.Event` as the stop flag (equivalent of `std::atomic<bool>`).
- `_step()` holds one iteration of work. **It must never block indefinitely**: use `timeout=` on buffer calls, or rely on `close()` to wake it.
- Python has no reliable destructor. Always call `stop()` explicitly (or use a `with` block on the owner).
- GIL: threads suit I/O, GPU launches and NumPy work (which releases the GIL). For CPU-heavy pure-Python work, use `multiprocessing` / `ProcessPoolExecutor` instead.

```python
# interface/workflow.py
from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod

from communication.buffers import BufferClosed
from communication.communication_module import CommunicationModule

logger = logging.getLogger(__name__)

class Workflow(ABC):
    def __init__(self, comm: CommunicationModule) -> None:
        self._comm = comm                       # non-owning reference
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

    def start(self) -> None:
        with self._lock:
            if self.is_running():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run, name=type(self).__name__, daemon=False
            )
            self._thread.start()

    def stop(self, timeout: float | None = None) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout)

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _run(self) -> None:
        try:
            while not self._stop_event.is_set():
                self._step()
        except BufferClosed:
            pass                                # normal shutdown path
        except Exception:
            logger.exception("%s stopped by an error", type(self).__name__)

    @abstractmethod
    def _step(self) -> None:
        """One iteration of work. Must not block forever."""
```

```python
# workflow/workflow1.py
from __future__ import annotations

from queue import Empty

from interface.workflow import Workflow

class Workflow1(Workflow):
    def _step(self) -> None:
        try:
            frame_id = self._comm.new_frame_ids.pop(timeout=0.1)
        except Empty:
            return                              # re-check stop flag

        result = self._do_something(frame_id)
        self._comm.workflow1_result.push(result)

    def _do_something(self, frame_id: int) -> int:
        return frame_id
```

## 2.2. Communication Pattern

- Use a ring buffer or a channel (queue) for inter-thread communication.
- Use a single shared `CommunicationModule` to manage communication buffers.
- All buffers are thread-safe. Do not share a plain `list` / `deque` between threads.
- Common buffer API:

| Method | Behavior |
| --- | --- |
| `push(value, timeout=None)` | Add an item. Raises `queue.Full` on timeout, `BufferClosed` if closed |
| `pop(timeout=None)` | Take the oldest item. Raises `queue.Empty` on timeout, `BufferClosed` if closed and empty |
| `close()` | Wake all waiting threads. Remaining items can still be popped |
| `clear()` | Remove all items and reopen the buffer |
| `len(buf)` | Current number of items |

| C++ | Python | When to use |
| --- | --- | --- |
| `BlockingRingBuffer` | `BlockingRingBuffer` | Every item matters; producer waits when full |
| `OverwriteRingBuffer` | `OverwriteRingBuffer` | Only the latest data matters; oldest is dropped when full |
| `Channel` | `Channel` | Unbounded queue with close/stop; supports `for item in channel` |

```python
# communication/buffers.py
from __future__ import annotations

import threading
from collections import deque
from queue import Empty, Full
from typing import Generic, Iterator, TypeVar

T = TypeVar("T")

class BufferClosed(Exception):
    """Push to a closed buffer, or pop from a closed and drained buffer."""

class BlockingRingBuffer(Generic[T]):
    """Fixed capacity. push() blocks while full, pop() blocks while empty."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._capacity = capacity
        self._items: deque[T] = deque()
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._not_full = threading.Condition(self._lock)
        self._closed = False

    def push(self, value: T, timeout: float | None = None) -> None:
        with self._lock:
            ok = self._not_full.wait_for(
                lambda: self._closed or len(self._items) < self._capacity, timeout
            )
            if not ok:
                raise Full
            if self._closed:
                raise BufferClosed
            self._items.append(value)
            self._not_empty.notify()

    def pop(self, timeout: float | None = None) -> T:
        with self._lock:
            ok = self._not_empty.wait_for(
                lambda: self._closed or len(self._items) > 0, timeout
            )
            if not ok:
                raise Empty
            if not self._items:
                raise BufferClosed
            value = self._items.popleft()
            self._not_full.notify()
            return value

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._not_empty.notify_all()
            self._not_full.notify_all()

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self._closed = False
            self._not_full.notify_all()

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

class OverwriteRingBuffer(Generic[T]):
    """Fixed capacity. push() never blocks; the oldest item is dropped when full."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._items: deque[T] = deque(maxlen=capacity)
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._closed = False

    def push(self, value: T, timeout: float | None = None) -> None:
        with self._lock:
            if self._closed:
                raise BufferClosed
            self._items.append(value)           # deque(maxlen) drops the oldest
            self._not_empty.notify()

    def pop(self, timeout: float | None = None) -> T:
        with self._lock:
            ok = self._not_empty.wait_for(
                lambda: self._closed or len(self._items) > 0, timeout
            )
            if not ok:
                raise Empty
            if not self._items:
                raise BufferClosed
            return self._items.popleft()

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._not_empty.notify_all()

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self._closed = False

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

class Channel(Generic[T]):
    """Unbounded queue. pop() blocks until data arrives or the channel is closed."""

    def __init__(self) -> None:
        self._items: deque[T] = deque()
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._closed = False

    def push(self, value: T, timeout: float | None = None) -> None:
        with self._lock:
            if self._closed:
                raise BufferClosed
            self._items.append(value)
            self._not_empty.notify()

    def pop(self, timeout: float | None = None) -> T:
        with self._lock:
            ok = self._not_empty.wait_for(
                lambda: self._closed or len(self._items) > 0, timeout
            )
            if not ok:
                raise Empty
            if not self._items:
                raise BufferClosed
            return self._items.popleft()

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._not_empty.notify_all()

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self._closed = False

    def __iter__(self) -> Iterator[T]:
        """Yield items until the channel is closed and drained."""
        while True:
            try:
                yield self.pop()
            except BufferClosed:
                return

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)
```

## 2.3. Communication Module

- Use a single `CommunicationModule` instance for communication between workflow threads.
- `CommunicationModule` is owned by the main application.
- Pass it to workflow threads as a **non-owning reference**.
- Workflow threads must **not** call `clear()` or `close()` on it, or replace its buffers. Only the owner does.
- Call `clear()` before starting the workflow threads.
- On shutdown, call `close()` **before** stopping the threads, so any thread blocked in `push`/`pop` wakes up.

```python
# communication/communication_module.py
from __future__ import annotations

from dataclasses import dataclass, field

from .buffers import BlockingRingBuffer, Channel, OverwriteRingBuffer

@dataclass
class CommunicationModule:
    new_frame_ids: Channel[int] = field(default_factory=Channel)
    workflow1_result: BlockingRingBuffer[int] = field(
        default_factory=lambda: BlockingRingBuffer(128)
    )
    workflow2_result: OverwriteRingBuffer[int] = field(
        default_factory=lambda: OverwriteRingBuffer(128)
    )
    workflow3_result: Channel[int] = field(default_factory=Channel)

    def _buffers(self) -> tuple:
        return (
            self.new_frame_ids,
            self.workflow1_result,
            self.workflow2_result,
            self.workflow3_result,
        )

    def clear(self) -> None:
        """Call before the workflow threads start."""
        for buf in self._buffers():
            buf.clear()

    def close(self) -> None:
        """Call before the workflow threads stop."""
        for buf in self._buffers():
            buf.close()
```

```python
# app.py
from __future__ import annotations

from communication.communication_module import CommunicationModule
from interface.workflow import Workflow
from workflow.workflow1 import Workflow1
from workflow.workflow2 import Workflow2

class Application:
    def __init__(self) -> None:
        self._comm = CommunicationModule()               # owner
        self._workflows: list[Workflow] = [
            Workflow1(self._comm),                       # non-owning reference
            Workflow2(self._comm),
        ]

    def start(self) -> None:
        self._comm.clear()
        for wf in self._workflows:
            wf.start()

    def stop(self) -> None:
        self._comm.close()                               # wake blocked threads
        for wf in reversed(self._workflows):
            wf.stop()

    def __enter__(self) -> "Application":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

if __name__ == "__main__":
    with Application() as app:
        ...
```