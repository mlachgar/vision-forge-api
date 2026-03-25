# Vision Forge API

Vision Forge API is a FastAPI service for image tagging powered by SigLIP.
It scores uploaded images against config-driven canonical tag sets and
prediction profiles, with bearer API key authentication, a small admin
surface for key management and config reloads, and an async batch job path for
high-volume photo analysis.

## Features

- FastAPI + Uvicorn service
- SigLIP image/text scoring
- Config-driven tag sets, profiles, and prompt templates
- Bearer API key auth with `admin` and `predict` roles
- `/predict` scoring with cached text embeddings
- `/predict/jobs` batch submission with in-memory micro-batching
- Prompt-level reranking for canonical tags
- Light set-balancing when a profile spans multiple tag sets
- Startup warmup for lower first-request latency
- Admin endpoints for API key CRUD and live config reload
- Docker image variants for CPU and GPU builds

## Runtime Layout

The container uses two directories:

### `/config`

Configuration directory. The image ships with sample config files here at build
time, and the path can be overridden with `VISION_FORGE_CONFIG_DIR`.

Expected files:

- `auth.yaml`
- `settings.yaml`
- `tag_sets.yaml`
- `profiles.yaml`
- `prompts.yaml`

Typical responsibilities:

- `auth.yaml` defines token prefix, token length, and default roles
- `settings.yaml` defines the app name, prediction limits, embedding location,
  model cache location, and SigLIP model id
- `tag_sets.yaml` defines canonical tag groups
- `profiles.yaml` defines prediction profiles and which tag sets they use
- `prompts.yaml` defines prompt templates for canonical tags

### `/data`

Writable runtime storage. The path can be overridden with
`VISION_FORGE_DATA_DIR`.

Expected content:

- `api_keys.json` - persisted API keys and roles used by the auth cache
- `embeddings/text_embeddings.json` - cached text embeddings for canonical tags
- `embeddings/metadata.json` - embedding cache metadata
- `model_cache/` - Hugging Face / Transformers model cache

The service will create or refresh missing embedding cache entries on startup
when needed.

The prediction model is loaded during startup and the app warms the prediction
pipeline before serving requests. Batch jobs are queued in memory, so they are
fast for live processing but are not durable across process restarts.
Finished batch jobs are retained for 15 minutes, and the in-memory job store
is capped at 5,000 retained items to prevent unbounded growth.

## API Surface

- `GET /health`
- `GET /tag-sets`
- `GET /profiles`
- `POST /predict`
- `POST /predict/jobs`
- `GET /predict/jobs/{job_id}`
- `DELETE /predict/jobs/{job_id}`
- `GET /admin/api-keys`
- `POST /admin/api-keys`
- `PATCH /admin/api-keys/{name}`
- `DELETE /admin/api-keys/{name}`
- `POST /admin/reload`

The prediction endpoint expects a multipart image upload and supports these
query parameters:

- `limit`
- `min_score`
- `profile`
- `tag_sets`
- `extra_tags`

Returned scores are normalized to the `0.0..1.0` range.

The batch job endpoint accepts multiple uploaded images in one request and
returns a `202 Accepted` response with a job id. Poll the job endpoint until
the job reaches a terminal state:

- `done`
- `partial`
- `failed`
- `canceled`

## Prediction Behavior

The prediction pipeline does the following:

1. Encodes the image once with SigLIP
2. Scores candidate canonical tags and any extra tags
3. Reranks the strongest canonical candidates using the best matching prompt
   similarity
4. Filters by `min_score`
5. Sorts by score descending
6. Applies the requested `limit`
7. Balances results lightly across tag sets when a profile spans multiple sets

For batch jobs, the app groups multiple photos into small server-side batches
to reduce per-photo overhead and improve throughput. The client should use the
job endpoint for large imports and keep `POST /predict` for single-image
requests.

## Docker Images

Published variants:

- `cpu-lite`
- `cpu-full`
- `gpu-lite`
- `gpu-full`

Release builds publish both floating variant tags and versioned tags. For
example, a release such as `v1.2.3` publishes:

- `1.2.3-cpu-lite`
- `1.2.3-cpu-full`
- `1.2.3-gpu-lite`
- `1.2.3-gpu-full`
- `cpu-lite`
- `cpu-full`
- `gpu-lite`
- `gpu-full`
- `latest` for `cpu-full` only

## Usage

Run the container with runtime data mounted at `/data`:

```bash
docker run --rm -it \
  -p 8000:8000 \
  -v "$PWD/data:/data" \
  -e VISION_FORGE_DEVICE=cpu \
  mlachgar/vision-forge-api:cpu-full
```

If you want to override the bundled config, mount your own directory at
`/config` and point `VISION_FORGE_CONFIG_DIR` to it:

```bash
docker run --rm -it \
  -p 8000:8000 \
  -v "$PWD/config:/config:ro" \
  -v "$PWD/data:/data" \
  -e VISION_FORGE_CONFIG_DIR=/config \
  -e VISION_FORGE_DEVICE=cpu \
  mlachgar/vision-forge-api:cpu-full
```

Batch job example:

```bash
curl -s -X POST \
  "http://127.0.0.1:8000/predict/jobs?tag_sets=animals&limit=10" \
  -H "Authorization: Bearer vfk_predict_token_87654321fedcba" \
  -F "files=@samples/cat.jpg" \
  -F "files=@samples/snail.jpg"
```

Poll the job:

```bash
curl -s \
  "http://127.0.0.1:8000/predict/jobs/<job_id>" \
  -H "Authorization: Bearer vfk_predict_token_87654321fedcba"
```
