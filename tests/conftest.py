from __future__ import annotations

import contextlib
import sys
import types
from types import SimpleNamespace


def _install_model_stubs() -> None:
    """Allow importing the app wiring without the heavy model dependencies."""

    if "torch" not in sys.modules:
        torch_stub = types.ModuleType("torch")

        def _device(name: str) -> SimpleNamespace:
            return SimpleNamespace(type=name)

        torch_stub.device = _device
        torch_stub.cuda = SimpleNamespace(is_available=lambda: False)
        torch_stub.no_grad = contextlib.nullcontext
        torch_stub.tensor = lambda data, **_: data
        torch_stub.stack = lambda values, dim=0: values
        torch_stub.matmul = lambda left, right: left
        torch_stub.empty = lambda *args, **kwargs: []
        torch_stub.float32 = "float32"

        nn_stub = types.ModuleType("torch.nn")
        functional_stub = types.ModuleType("torch.nn.functional")
        functional_stub.normalize = lambda value, dim=-1: value
        nn_stub.functional = functional_stub
        torch_stub.nn = nn_stub

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
                return []

            def get_text_features(self, *args, **kwargs):
                return []

        transformers_stub.SiglipModel = _StubModel
        transformers_stub.SiglipProcessor = _StubProcessor
        sys.modules["transformers"] = transformers_stub


_install_model_stubs()
