#!/usr/bin/env python3
"""Compatibility helpers for the archived TabPFN 2.5/ManyClass sensitivity run.

The persisted CSV under ``results/round4/raw`` is the canonical result used by
this article. These helpers support optional model re-execution with
``tabpfn==6.0.6`` without changing the recorded experimental configuration.
"""
from __future__ import annotations

import importlib.metadata
import importlib.util
import os
from pathlib import Path
from typing import Any

TABPFN_VERSION = "6.0.6"
TABPFN_RELEASE_COMMIT = "4103e892d8c3059183ac7c507b923924ac9b1091"
ASSET_NAME = "tabpfn_col_embedding.pt"
ASSET_RELATIVE_PATHS = (
    f"src/tabpfn/architectures/shared/{ASSET_NAME}",
    f"src/tabpfn/architectures/base/{ASSET_NAME}",
)


def _patch_huggingface_hub() -> None:
    import huggingface_hub

    if hasattr(huggingface_hub, "HfFolder"):
        return

    class HfFolder:
        path_token = str(Path.home() / ".cache" / "huggingface" / "token")

        @classmethod
        def get_token(cls) -> str | None:
            try:
                return huggingface_hub.get_token()
            except Exception:
                return None

        @classmethod
        def save_token(cls, token: str | None) -> None:
            if not token:
                return
            path = Path(cls.path_token)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(token, encoding="utf-8")

        @classmethod
        def delete_token(cls) -> None:
            Path(cls.path_token).unlink(missing_ok=True)

    huggingface_hub.HfFolder = HfFolder


def _patch_preprocessor_alias() -> None:
    import tabpfn.inference_config as inference_config

    target = "default_classifier_preprocessor_configs"
    if hasattr(inference_config, target):
        return
    for candidate in (
        "v2_5_classifier_preprocessor_configs",
        "v2_classifier_preprocessor_configs",
    ):
        if hasattr(inference_config, candidate):
            setattr(inference_config, target, getattr(inference_config, candidate))
            return
    raise RuntimeError("No compatible TabPFN classifier-preprocessor factory found")


def _validate_torch_asset(path: Path) -> dict[str, Any]:
    import torch

    if not path.is_file() or path.stat().st_size < 512:
        raise RuntimeError(f"Invalid TabPFN embedding asset: {path}")
    head = path.read_bytes()[:256].lstrip()
    if head.startswith((b"<", b"<!DOCTYPE", b"<?xml", b"version https://git-lfs")):
        raise RuntimeError(f"Downloaded object is not the embedding asset: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")

    def tensors(value: Any) -> list[Any]:
        if torch.is_tensor(value):
            return [value]
        if isinstance(value, dict):
            return [t for item in value.values() for t in tensors(item)]
        if isinstance(value, (list, tuple)):
            return [t for item in value for t in tensors(item)]
        return []

    items = tensors(payload)
    if not items or sum(int(t.numel()) for t in items) == 0:
        raise RuntimeError(f"Embedding asset contains no tensor data: {path}")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "tensor_count": len(items),
        "first_shape": tuple(items[0].shape),
    }


def ensure_column_embedding_asset() -> dict[str, Any]:
    """Restore the package-local embedding asset omitted by some 6.0.6 wheels."""
    import requests

    spec = importlib.util.find_spec("tabpfn.architectures.shared")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("Could not locate tabpfn.architectures.shared")
    destination = Path(next(iter(spec.submodule_search_locations))) / ASSET_NAME
    if destination.exists():
        try:
            return _validate_torch_asset(destination)
        except Exception:
            destination.unlink(missing_ok=True)

    destination.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for relative in ASSET_RELATIVE_PATHS:
        for host in (
            "https://raw.githubusercontent.com/PriorLabs/TabPFN",
            "https://media.githubusercontent.com/media/PriorLabs/TabPFN",
        ):
            url = f"{host}/{TABPFN_RELEASE_COMMIT}/{relative}"
            tmp = destination.with_suffix(".pt.downloading")
            tmp.unlink(missing_ok=True)
            try:
                with requests.get(url, stream=True, timeout=(30, 180)) as response:
                    response.raise_for_status()
                    with tmp.open("wb") as handle:
                        for chunk in response.iter_content(1 << 20):
                            if chunk:
                                handle.write(chunk)
                info = _validate_torch_asset(tmp)
                os.replace(tmp, destination)
                info["source"] = url
                return info
            except Exception as exc:
                last_error = exc
                tmp.unlink(missing_ok=True)
    raise RuntimeError("Could not restore the TabPFN column-embedding asset") from last_error


def prepare_tabpfn_environment(require_gpu: bool = True) -> dict[str, Any]:
    """Apply compatibility shims and configure the historical wrapper for CUDA."""
    import torch

    installed = importlib.metadata.version("tabpfn")
    if installed != TABPFN_VERSION:
        raise RuntimeError(
            f"Expected tabpfn=={TABPFN_VERSION}; found tabpfn=={installed}"
        )
    if require_gpu and not torch.cuda.is_available():
        raise RuntimeError("The archived ManyClass rerun requires a CUDA GPU")

    _patch_huggingface_hub()
    _patch_preprocessor_alias()
    asset = ensure_column_embedding_asset()

    from src.models import config

    config.TABPFN_CONFIG["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    config.TABPFN_CONFIG["ignore_pretraining_limits"] = True
    os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "1"

    return {
        "tabpfn_version": installed,
        "device": config.TABPFN_CONFIG["device"],
        "asset": asset,
    }


if __name__ == "__main__":
    print(prepare_tabpfn_environment())
