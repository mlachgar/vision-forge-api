#!/usr/bin/env python3
"""Precompute and persist text embeddings derived from SigLIP prompts."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vision_forge_api.catalog.service import TagCatalog
from vision_forge_api.config.loader import ConfigLoader
from vision_forge_api.config.schema import TagPrompt
from vision_forge_api.embeddings.store import EmbeddingStore, TEXT_EMBEDDING_FORMAT_VERSION
from vision_forge_api.siglip.service import SiglipService

LOGGER = logging.getLogger("precompute_embeddings")
DEFAULT_PROMPT = TagPrompt(template="An image of {tag}.", weight=1.0)


def _render_prompt(prompt: TagPrompt, tag: str) -> str:
    try:
        return prompt.template.format(tag=tag)
    except Exception:
        return prompt.template


def _compute_vector(siglip: SiglipService, prompts: tuple[TagPrompt, ...], tag: str) -> tuple[float, ...]:
    if not prompts:
        prompts = (DEFAULT_PROMPT,)
    texts: list[str] = []
    weights: list[float] = []
    for prompt in prompts:
        weight = float(prompt.weight)
        if weight <= 0:
            continue
        texts.append(_render_prompt(prompt, tag))
        weights.append(weight)
    if not texts:
        texts = [_render_prompt(DEFAULT_PROMPT, tag)]
        weights = [1.0]
    embeddings = siglip.encode_texts(texts)
    weight_tensor = torch.tensor(weights, dtype=torch.float32, device=siglip.device)
    averaged = F.normalize(
        (embeddings * weight_tensor.unsqueeze(1)).sum(dim=0) / weight_tensor.sum().clamp_min(1e-6),
        dim=-1,
    )
    return tuple(float(value) for value in averaged.cpu().tolist())


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate canonical tag embeddings using SigLIP prompts.")
    parser.add_argument("--config-dir", type=Path, default=Path("/config"), help="Path to the mounted /config directory")
    parser.add_argument("--device", default=None, help="Preferred device (e.g., cpu, cuda)")
    parser.add_argument("--embeddings-dir", type=Path, default=None, help="Override the directory where embeddings are persisted")
    parser.add_argument("--model-cache-dir", type=Path, default=None, help="Override the SigLIP cache directory")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    loader = ConfigLoader(args.config_dir)
    settings = loader.load_settings()
    overrides: dict[str, Path] = {}
    if args.embeddings_dir:
        overrides["embeddings_dir"] = args.embeddings_dir
    if args.model_cache_dir:
        overrides["model_cache_dir"] = args.model_cache_dir
    if overrides:
        settings = settings.model_copy(update=overrides)  # type: ignore[arg-type]
    catalog = TagCatalog(loader.load_tag_sets(), loader.load_profiles(), loader.load_prompts())
    siglip = SiglipService(settings.siglip_model_id, settings.model_cache_dir, device_hint=args.device)
    store = EmbeddingStore(settings.embeddings_dir)
    tags = catalog.canonical_tags()
    LOGGER.info("Precomputing embeddings for %d canonical tags", len(tags))
    payload: dict[str, tuple[float, ...]] = {}
    for tag in tags:
        prompts = catalog.prompts_for_tag(tag)
        vector = _compute_vector(siglip, prompts, tag)
        payload[tag] = vector
        LOGGER.debug("Computed embedding for tag %s (len=%d)", tag, len(vector))
    store.persist(
        payload,
        model_id=settings.siglip_model_id,
        format_version=TEXT_EMBEDDING_FORMAT_VERSION,
    )
    LOGGER.info("Persisted %d embeddings to %s", len(payload), settings.embeddings_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
