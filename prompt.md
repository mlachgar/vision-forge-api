You are a senior Python backend + MLOps engineer.

Your job is to help build a production-quality REST API application for image tagging/classification using a pretrained SigLIP model.


The overall project to build is the following:

========================
HIGH-LEVEL GOAL
========================

- Build a Docker-friendly Python REST API for image understanding:
  - tagging (v1)
  - future captioning / generation
- Use pretrained SigLIP (vision-language model)
- Score images against user-defined tag vocabularies
- Use latest stable libraries only
- Use Python 3.12 unless incompatible
- Pin exact versions after resolving compatibility

Docker image name:
vision-forge-api

========================
TECH STACK
========================

- FastAPI
- Uvicorn
- Pydantic v2
- PyTorch (CPU default)
- Hugging Face Transformers
- Pillow
- PyYAML
- pytest
- ruff
- mypy
- httpx

NO DATABASE in v1

========================
CORE PRODUCT DECISIONS
========================

- Auth: Bearer API key only
- No OAuth/OIDC
- No DB
- File-based persistence
- Dynamic API key management via API
- In-memory cache for tokens
- Mounted volumes REQUIRED
- Tag system = domain tag sets + profiles
- Model = SigLIP only
- Precomputed embeddings
- Clean architecture for future captioning

========================
AUTHENTICATION DESIGN
========================

Token transport:
Authorization: Bearer <token>

Storage model:

STATIC CONFIG (user-managed)
- /config/auth.yaml

DYNAMIC DATA (app-managed)
- /data/api_keys.json

Mounted volumes REQUIRED:
- /config
- /data

Runtime behavior:

On startup:
- load auth.yaml
- load api_keys.json
- merge into in-memory cache

Cache structure:
key_hash -> {name, roles, enabled}

On request:
- hash incoming token (sha256)
- lookup in memory
- authorize

NEVER read files per request

Security:
- NEVER store plaintext tokens
- store SHA256 hashes only
- return token only once
- use secure random tokens (prefix: vfk_...)

========================
API KEY MANAGEMENT
========================

ADMIN ONLY endpoints:

POST /admin/api-keys
- create key
- generate token
- hash + store in /data/api_keys.json
- return plaintext ONCE

GET /admin/api-keys
- list keys (NO raw tokens)

DELETE /admin/api-keys/{name}
- revoke

PATCH /admin/api-keys/{name}
- enable/disable

After any change:
- atomic write to file
- rebuild in-memory cache

File format (JSON only):

[
  {
    "name": "mobile-app",
    "key_hash": "sha256:...",
    "roles": ["predict"],
    "enabled": true
  }
]

========================
PERSISTENCE DESIGN
========================

Mounted volumes REQUIRED:

docker run:
-v ./config:/config
-v ./data:/data

Structure:

/config
  auth.yaml
  settings.yaml
  tag_sets.yaml
  profiles.yaml
  prompts.yaml

/data
  api_keys.json
  embeddings/
  model_cache/

Rules:
- NEVER write to /config
- ALWAYS write runtime data to /data
- use atomic file writes

========================
IN-MEMORY CACHE
========================

- O(1) lookup
- built at startup
- refreshed on:
  - key changes
  - config reload

Prefer full rebuild over partial mutation

========================
API DESIGN
========================

POST /predict

- multipart/form-data:
  file=image

- query params:
  limit: int | optional
  min_score: float | optional
  profile
  tag_sets
  extra_tags

Processing:
1. score all tags
2. filter by min_score
3. sort desc
4. apply limit

Return:
{
  "tags": [
    {"label": "...", "score": 0.91}
  ],
  "meta": {...}
}

Other endpoints:
GET /health
GET /tag-sets
GET /profiles
POST /admin/reload

========================
ML DESIGN (SigLIP)
========================

- Use SigLIP ONLY
- shared embedding space
- precompute text embeddings
- store in /data/embeddings

Inference:
- encode image once
- compare to tag embeddings

Tags:
- canonical tags
- prompts per tag
- tag sets
- profiles (group of sets)

Post-processing:
- deduplicate
- keep best canonical label

========================
DOCKER DESIGN
========================

ONE Dockerfile ONLY

Path:
docker/Dockerfile

Build args:
- DEVICE=cpu|gpu
- BUNDLE_MODEL=true|false

Variants:

cpu-lite   => cpu + no model
cpu-full   => cpu + bundled model
gpu-lite   => gpu + no model
gpu-full   => gpu + bundled model

Tags:

vision-forge-api:cpu-lite
vision-forge-api:cpu-full
vision-forge-api:gpu-lite
vision-forge-api:gpu-full

Versioned:

vision-forge-api:1.0.0-cpu-lite
vision-forge-api:1.0.0-cpu-full
vision-forge-api:1.0.0-gpu-lite
vision-forge-api:1.0.0-gpu-full

latest:
→ MUST point to cpu-full only

Definitions:
- lite = no model baked
- full = model baked

Requirements:
- non-root user
- port 8000
- deterministic startup
- documented volumes
- minimal duplication

========================
CI/CD (GitHub Actions)
========================

Workflows:

.github/workflows/ci.yml
.github/workflows/docker-publish.yml

CI:
- lint
- format check
- type check
- tests
- optional docker smoke build

Docker publish:
- trigger on tag vX.Y.Z
- matrix build (4 variants)
- use buildx
- push to Docker Hub

Tags:
- versioned (all 4)
- latest → cpu-full only

Secrets:
- DOCKERHUB_USERNAME
- DOCKERHUB_TOKEN
- DOCKERHUB_NAMESPACE

Optional:
- edge tags for main branch

Security:
- no secrets in logs
- pinned actions
- least privilege

========================
TESTING
========================

- mock SigLIP
- test auth
- test API keys
- test cache reload
- test predict filters
- test invalid inputs
- test profile resolution

========================
DEVELOPER EXPERIENCE
========================

Makefile:

- install
- run
- test
- lint
- format
- typecheck
- precompute-embeddings
- docker-build-cpu-lite
- docker-build-cpu-full
- docker-build-gpu-lite
- docker-build-gpu-full

README must include:
- architecture
- endpoints
- auth
- config
- docker usage
- CI/CD
- release process
- example curl
- docker pull examples

========================
QUALITY BAR
========================

- FULL working project
- NO placeholders
- strong typing
- clean architecture
- production-ready v1
- maintainable
- explicit over magic

========================
EXECUTION MODE
========================

Use a practical step sequence such as:
- Step 1: dependency selection and project skeleton
- Step 2: config models and loaders
- Step 3: auth core and token cache
- Step 4: FastAPI app bootstrap and health endpoint
- Step 5: tags, profiles, and prompt catalog
- Step 6: SigLIP abstraction and embedding generation
- Step 7: predict endpoint and scoring pipeline
- Step 8: admin API key endpoints
- Step 9: reload flow and persistence hardening
- Step 10: tests
- Step 11: Docker
- Step 12: CI/CD
- Step 13: README and Makefile

ths steps 1 to 9 are done
