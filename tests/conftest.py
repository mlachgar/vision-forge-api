from __future__ import annotations

import contextlib
import sys
import types
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class _Device:
    type: str

    def __str__(self) -> str:
        return self.type


class _Tensor:
    def __init__(self, value: Any):
        self._array = np.asarray(value)

    @property
    def T(self) -> "_Tensor":
        return _Tensor(self._array.T)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self._array.shape)

    @property
    def ndim(self) -> int:
        return int(self._array.ndim)

    def unsqueeze(self, axis: int) -> "_Tensor":
        return _Tensor(np.expand_dims(self._array, axis))

    def squeeze(self, axis: int | None = None) -> "_Tensor":
        return _Tensor(np.squeeze(self._array, axis=axis))

    def sum(self, dim: int | None = None) -> "_Tensor":
        return _Tensor(self._array.sum(axis=dim))

    def numel(self) -> int:
        return int(self._array.size)

    def cpu(self) -> "_Tensor":
        return self

    def to(self, *_args: Any, **_kwargs: Any) -> "_Tensor":
        return self

    def item(self) -> float:
        return float(self._array.item())

    def clamp_min(self, minimum: float) -> "_Tensor":
        return _Tensor(np.maximum(self._array, minimum))

    def tolist(self) -> list[Any]:
        return self._array.tolist()

    def __iter__(self):
        for item in self._array:
            yield _Tensor(item) if np.ndim(item) else item.item()

    def __getitem__(self, item):
        value = self._array[item]
        return _Tensor(value) if np.ndim(value) else value.item()

    def __len__(self) -> int:
        return len(self._array)

    def __mul__(self, other: Any) -> "_Tensor":
        return _Tensor(self._array * _unwrap(other))

    def __rmul__(self, other: Any) -> "_Tensor":
        return _Tensor(_unwrap(other) * self._array)

    def __truediv__(self, other: Any) -> "_Tensor":
        return _Tensor(self._array / _unwrap(other))

    def __repr__(self) -> str:
        return repr(self.tolist())

    def __array__(self, dtype=None):
        return np.asarray(self._array, dtype=dtype)


def _unwrap(value: Any) -> Any:
    return value._array if isinstance(value, _Tensor) else value


def _tensor(value: Any, **_: Any) -> _Tensor:
    return _Tensor(value)


def _stack(values: list[_Tensor] | tuple[_Tensor, ...], dim: int = 0) -> _Tensor:
    arrays = [_unwrap(value) for value in values]
    return _Tensor(np.stack(arrays, axis=dim))


def _matmul(left: Any, right: Any) -> _Tensor:
    return _Tensor(np.matmul(_unwrap(left), _unwrap(right)))


def _max(value: Any) -> _Tensor:
    return _Tensor(np.max(_unwrap(value)))


def _ones(*shape: int, **_: Any) -> _Tensor:
    return _Tensor(np.ones(shape))


def _allclose(left: Any, right: Any, atol: float = 1e-8, rtol: float = 1e-5) -> bool:
    return bool(
        np.allclose(_unwrap(left), _unwrap(right), atol=atol, rtol=rtol, equal_nan=True)
    )


def _empty(*shape: int, **_: Any) -> _Tensor:
    if not shape:
        shape = (0,)
    return _Tensor(np.empty(shape))


def _normalize(value: Any, dim: int = -1) -> _Tensor:
    array = np.asarray(_unwrap(value), dtype=float)
    norm = np.linalg.norm(array, axis=dim, keepdims=True)
    norm = np.where(norm == 0, 1.0, norm)
    return _Tensor(array / norm)


def _install_model_stubs() -> None:
    """Allow importing the app wiring without the heavy model dependencies."""

    if "torch" not in sys.modules:
        torch_stub = types.ModuleType("torch")
        torch_stub.device = lambda name: _Device(type=name)
        torch_stub.cuda = types.SimpleNamespace(is_available=lambda: False)
        torch_stub.no_grad = contextlib.nullcontext
        torch_stub.tensor = _tensor
        torch_stub.stack = _stack
        torch_stub.matmul = _matmul
        torch_stub.max = _max
        torch_stub.ones = _ones
        torch_stub.allclose = _allclose
        torch_stub.empty = _empty
        torch_stub.float32 = np.float32

        nn_stub = types.ModuleType("torch.nn")
        functional_stub = types.ModuleType("torch.nn.functional")
        functional_stub.normalize = _normalize
        nn_stub.functional = functional_stub
        torch_stub.nn = nn_stub
        linalg_stub = types.SimpleNamespace(
            vector_norm=lambda value, dim=None: _Tensor(
                np.linalg.norm(_unwrap(value), axis=dim)
            )
        )
        torch_stub.linalg = linalg_stub

        sys.modules["torch"] = torch_stub
        sys.modules["torch.nn"] = nn_stub
        sys.modules["torch.nn.functional"] = functional_stub

    if "transformers" not in sys.modules:
        transformers_stub = types.ModuleType("transformers")

        class _StubProcessor:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                return cls()

        class _StubModel:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                return cls()

            def to(self, *args, **kwargs):
                return self

            def eval(self):
                return self

            def get_image_features(self, *args, **kwargs):
                return _tensor([[1.0, 0.0]])

            def get_text_features(self, *args, **kwargs):
                return _tensor([[1.0, 0.0]])

        transformers_stub.SiglipModel = _StubModel
        transformers_stub.SiglipProcessor = _StubProcessor
        sys.modules["transformers"] = transformers_stub


_install_model_stubs()
