# syntax=docker/dockerfile:1.7

ARG PYTHON_VERSION=3.12
ARG DEVICE=cpu
ARG BUNDLE_MODEL=false
ARG SIGLIP_MODEL_ID=google/siglip-base-patch16-224

FROM python:${PYTHON_VERSION}-slim AS runtime

ARG DEVICE
ARG BUNDLE_MODEL
ARG SIGLIP_MODEL_ID

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH=/home/app/.local/bin:${PATH} \
    VISION_FORGE_DEVICE=${DEVICE}

WORKDIR /app

RUN addgroup --system app \
    && adduser --system --home /home/app --ingroup app app

COPY pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts
COPY config/. /config/

RUN mkdir -p /config /data/embeddings /data/model_cache \
    && chown -R app:app /app /config /data

USER app

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install --user . \
    && if [ "$BUNDLE_MODEL" = "true" ]; then \
         PYTHONPATH=src python scripts/bundle_model.py \
           --model-id "$SIGLIP_MODEL_ID" \
           --model-cache-dir /data/model_cache \
           --device cpu; \
       fi
EXPOSE 8000

CMD ["vision-forge-api"]
