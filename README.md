# Vision Forge API

Production-ready FastAPI service for image tagging using SigLIP.

## Status

The API is operational and includes:
- API key auth with role-based access (`admin`, `predict`)
- Config-driven tag catalog, profiles, and prompt templates
- SigLIP-based image/text scoring with precomputed text embeddings
- Predict reranking (prompt-level) and light per-set balancing for broad profiles
- Admin endpoints for API key management and live config reload

## Overview

- **Tech stack:** FastAPI, PyTorch, Transformers (Hugging Face), Pydantic v2, Uvicorn
- **Purpose:** Score uploaded images against canonical tag vocabularies and profiles bundled into the image.

## Repository Layout

- `/config` (read-only in the image): `auth.yaml`, `settings.yaml`, `tag_sets.yaml`, `profiles.yaml`, `prompts.yaml`
- `/data` (read/write): `api_keys.json`, `embeddings/`, `model_cache/`
- `/samples`: 20 sample JPG files for quick manual testing
- `docker/Dockerfile`: image build definition

## Sample Config And Data

- `config/README.md` documents seeded demo tokens:
  - admin token: `vfk_admin_token_12345678abcdef12`
  - predict token: `vfk_predict_token_87654321fedcba`
- `data/api_keys.json` includes hashed API keys for those demo tokens.
- `data/embeddings/text_embeddings.json` stores precomputed text embeddings.
- `data/model_cache/` is used by Transformers model downloads.

## Installation (Non-Docker)

- Use Python 3.12.
- Install package:

```bash
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install --force-reinstall --no-deps .
python3 -m pip install ".[dev]"
```

## Prediction Notes

- Current taxonomy is large (100+ tags per set; ~1500+ canonical tags total).
- `config/prompts.yaml` contains prompt entries for all canonical tags.
- Text embeddings are cached in `data/embeddings/text_embeddings.json`.
- On startup, the embedding cache is loaded and auto-refreshed if model id/format metadata is stale.
- The SigLIP model is loaded during startup.
- Bundled `cpu-full` images load SigLIP from the pre-populated local cache in `/data/model_cache`.
- `/predict` uses two ranking stages:
  - coarse rank with cached per-tag embeddings
  - rerank top canonical candidates with prompt-level max similarity
- For the broad default profile, a light per-set balancing penalty is applied to reduce single-set dominance.
- Scores are normalized to `0.0..1.0` before filtering and returning. Default `min_score` is configured in `config/settings.yaml`.
- `/predict/jobs` accepts multiple photos and processes them through a small in-memory micro-batch queue.
  - use it when you want higher throughput on photo bursts
  - keep `/predict` for single-image, immediate responses
  - queued jobs are not persisted across process restarts in this version
  - finished jobs are retained for 15 minutes, then cleaned up
  - the in-memory job store is capped at 5,000 retained items

## Helper Scripts

- `scripts/validate_config.py`: validates config structure.
- `scripts/precompute_embeddings.py`: recomputes and persists text embeddings.
- `scripts/download_model.py`: pre-downloads model artifacts into cache.
- `scripts/test_embeddings.py`: checks embedding coverage against canonical tags.
- `scripts/docker_smoke_test.py`: runs the Docker smoke e2e checks for `/health` and `/predict`.

Pytest markers:

- `integration`: HTTP-level tests that exercise the API with bundled config, runtime data, and sample images.

## Docker Workflow

Build image:

```bash
docker build --no-cache -t vision-forge-api:dev -f docker/Dockerfile .
```

Run API container:

```bash
docker run --rm -it \
  -p 8000:8000 \
  -v "$PWD/data:/data" \
  -v "$PWD/samples:/samples:ro" \
  -e VISION_FORGE_DEVICE=cpu \
  vision-forge-api:dev
```

## CI/CD

This repository includes four GitHub Actions workflows:

- `.github/workflows/ci.yml`
- `.github/workflows/docker-publish.yml`
- `.github/workflows/docker-smoke.yml`
- `.github/workflows/published-docker-test.yml`

Badge templates (replace `<owner>` and `<repo>`):

- `![CI](https://github.com/<owner>/<repo>/actions/workflows/ci.yml/badge.svg)`
- `![Docker Publish](https://github.com/<owner>/<repo>/actions/workflows/docker-publish.yml/badge.svg)`

### CI workflow (`ci.yml`)

Triggers:

- Pull requests
- Pushes to `main`
- Manual run (`workflow_dispatch`)

What it runs:

1. Python `3.12.13` setup with pip caching
2. Dependency install
3. Ruff lint (`ruff check`)
4. Ruff format check (`ruff format --check`)
5. Mypy type check
6. Config validation (`scripts/validate_config.py`)
7. Test suite (`pytest`) with artifact upload
8. Coverage XML artifact for the full suite
9. SonarQube Cloud scan
10. Dependency audit (`pip-audit`, non-blocking)

Required CI secret for the Sonar scan:

- `SONAR_TOKEN`

### Docker build workflow (`docker-smoke.yml`)

Triggers:

- Automatically after `CI` completes successfully (`workflow_run`)
- Manual run (`workflow_dispatch`)

What it runs:

1. Checkout the exact commit SHA from the completed CI run
2. Build `cpu-lite` image from `docker/Dockerfile` (no push)
3. Uses Buildx + GHA cache

Dependabot branches are skipped to avoid duplicate Docker build runs on
dependency update PRs.

### Published Docker Test workflow (`published-docker-test.yml`)

Triggers:

- Automatically after `Docker Publish` completes successfully (`workflow_run`)
- Manual run (`workflow_dispatch`)

What it runs:

1. Checkout the exact commit SHA from the completed publish run
2. Pull the published `cpu-full` Docker image
3. Runs a container smoke check against `/health`
4. Runs one authenticated `/predict` request with a sample image from `samples/`

The smoke helper will bootstrap a local `data/` tree with demo API keys if the
checkout does not already include one.

### Docker publish workflow (`docker-publish.yml`)

Triggers:

- Push tags matching `v*.*.*` (release publication)
- Only publishes when the tagged commit belongs to `main`

Build/publish behavior:

- Uses one Dockerfile: `docker/Dockerfile`
- Uses a matrix for 4 variants with explicit build args:
  - `cpu-lite` => `DEVICE=cpu`, `BUNDLE_MODEL=false`
  - `cpu-full` => `DEVICE=cpu`, `BUNDLE_MODEL=true`
  - `gpu-lite` => `DEVICE=gpu`, `BUNDLE_MODEL=false`
  - `gpu-full` => `DEVICE=gpu`, `BUNDLE_MODEL=true`
- Publishes to Docker Hub with configurable namespace
- Adds OCI metadata labels
- Uses Buildx + GHA cache
- Runs a non-blocking Trivy scan on `cpu-full`

Tag rules:

- Release tag `v1.0.0` publishes:
  - `1.0.0-cpu-lite`, `1.0.0-cpu-full`, `1.0.0-gpu-lite`, `1.0.0-gpu-full`
  - `cpu-lite`, `cpu-full`, `gpu-lite`, `gpu-full`
- `latest` is published only for `cpu-full`

## Docker Hub Configuration

Set these repository settings before publishing:

- Secret: `DOCKERHUB_USERNAME`
- Secret: `DOCKERHUB_TOKEN`
- Variable: `DOCKERHUB_NAMESPACE` (optional; falls back to `DOCKERHUB_USERNAME`)

Image repository name is fixed to `vision-forge-api`.

## Release Process

1. Merge your changes to `main`
2. Ensure `pyproject.toml` version matches the intended release
3. Create and push a Git tag, for example:

```bash
git tag v1.0.0
git push origin v1.0.0
```

4. `docker-publish.yml` builds and publishes all four image variants
5. `latest` is updated to the `cpu-full` release image only
6. Versioned tags (for example `1.0.0-cpu-full`) are treated as immutable by convention

## Pull Examples

Replace `<namespace>` with your Docker Hub namespace.

```bash
docker pull <namespace>/vision-forge-api:latest
docker pull <namespace>/vision-forge-api:cpu-lite
docker pull <namespace>/vision-forge-api:cpu-full
docker pull <namespace>/vision-forge-api:gpu-lite
docker pull <namespace>/vision-forge-api:gpu-full
```

`latest` always maps to `cpu-full`.

## API Endpoints

- `GET /health`
- `GET /tag-sets`
- `GET /profiles`
- `POST /predict` (requires `predict` role)
- `GET /admin/api-keys` (requires `admin` role)
- `POST /admin/api-keys` (requires `admin` role)
- `PATCH /admin/api-keys/{name}` (requires `admin` role)
- `DELETE /admin/api-keys/{name}` (requires `admin` role)
- `POST /admin/reload` (requires `admin` role)

## Localhost Docker Test Commands

Open a second terminal while the container is running and execute:

```bash
export BASE_URL="http://127.0.0.1:8000"
export ADMIN_TOKEN="vfk_admin_token_12345678abcdef12"
export PREDICT_TOKEN="vfk_predict_token_87654321fedcba"
```

Health:

```bash
curl -s "$BASE_URL/health"
```

Catalog:

```bash
curl -s "$BASE_URL/tag-sets"
curl -s "$BASE_URL/profiles"
```

Predict using the default profile:

```bash
curl -s -X POST \
  "$BASE_URL/predict?limit=10" \
  -H "Authorization: Bearer $PREDICT_TOKEN" \
  -F "file=@samples/snail.jpg"
```

Predict with explicit tag set scope (usually cleaner than full default):

```bash
curl -s -X POST \
  "$BASE_URL/predict?tag_sets=animals&limit=10" \
  -H "Authorization: Bearer $PREDICT_TOKEN" \
  -F "file=@samples/cat.jpg"
```

Predict with extra ad-hoc tags:

```bash
curl -s -X POST \
  "$BASE_URL/predict?tag_sets=animals&extra_tags=pet,macro_photo&limit=10" \
  -H "Authorization: Bearer $PREDICT_TOKEN" \
  -F "file=@samples/snail.jpg"
```

Admin list API keys:

```bash
curl -s "$BASE_URL/admin/api-keys" \
  -H "Authorization: Bearer $ADMIN_TOKEN"
```

Admin create API key:

```bash
curl -s -X POST "$BASE_URL/admin/api-keys" \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"name":"local-test-key","roles":["predict"],"enabled":true}'
```

Admin disable API key:

```bash
curl -s -X PATCH "$BASE_URL/admin/api-keys/local-test-key" \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"enabled":false}'
```

Admin delete API key:

```bash
curl -i -X DELETE "$BASE_URL/admin/api-keys/local-test-key" \
  -H "Authorization: Bearer $ADMIN_TOKEN"
```

Reload config/state:

```bash
curl -s -X POST "$BASE_URL/admin/reload" \
  -H "Authorization: Bearer $ADMIN_TOKEN"
```

## Troubleshooting

- If predictions are unexpectedly empty, lower `min_score` only if you set one explicitly. Omitting `min_score` keeps the default filter at `0.0`, which does not drop any normalized scores.
- After editing `config/tag_sets.yaml` or `config/prompts.yaml`, rerun:

```bash
PYTHONPATH=src ./.venv/bin/python scripts/precompute_embeddings.py \
  --config-dir config \
  --embeddings-dir data/embeddings \
  --model-cache-dir data/model_cache
```

- Validate embedding coverage:

```bash
PYTHONPATH=src ./.venv/bin/python scripts/test_embeddings.py
```
