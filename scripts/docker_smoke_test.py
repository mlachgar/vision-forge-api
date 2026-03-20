#!/usr/bin/env python3
"""Run a minimal Docker-based smoke test against the built API image."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Start the API container and verify /health and /predict."
    )
    parser.add_argument(
        "--image",
        default="vision-forge-api:ci-smoke",
        help="Docker image tag to run",
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="Base URL exposed by the container",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("config"),
        help="Host path mounted into /config",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Host path mounted into /data",
    )
    parser.add_argument(
        "--image-path",
        type=Path,
        default=Path("samples/cat.jpg"),
        help="Sample image to upload to /predict",
    )
    parser.add_argument(
        "--predict-token",
        default="vfk_predict_token_87654321fedcba",
        help="Bearer token used for the predict request",
    )
    parser.add_argument(
        "--profile",
        default="default",
        help="Profile query parameter used for /predict",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Prediction limit used for /predict",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=300,
        help="Maximum time to wait for the service to become healthy",
    )
    parser.add_argument(
        "--poll-interval-seconds",
        type=int,
        default=5,
        help="Delay between readiness checks",
    )
    return parser.parse_args()


def _run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=check, text=True, capture_output=True)


def _docker_logs(container_id: str) -> None:
    result = _run(["docker", "logs", container_id], check=False)
    if result.stdout:
        print(result.stdout, file=sys.stderr, end="")
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="")


def _start_container(args: argparse.Namespace) -> str:
    command = [
        "docker",
        "run",
        "-d",
        "--rm",
        "-p",
        "8000:8000",
        "-e",
        "VISION_FORGE_CONFIG_DIR=/config",
        "-e",
        "VISION_FORGE_DATA_DIR=/data",
        "-e",
        "VISION_FORGE_DEVICE=cpu",
        "-v",
        f"{args.config_dir.resolve()}:/config:ro",
        "-v",
        f"{args.data_dir.resolve()}:/data",
        args.image,
    ]
    result = _run(command)
    return result.stdout.strip()


def _stop_container(container_id: str) -> None:
    _run(["docker", "stop", container_id], check=False)


def _wait_for_health(
    base_url: str, timeout_seconds: int, poll_interval_seconds: int
) -> dict[str, object]:
    deadline = time.time() + timeout_seconds
    last_error: Exception | None = None
    health_url = urllib.parse.urljoin(base_url, "/health")
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(health_url, timeout=10) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if payload.get("status") == "ok":
                return payload
            last_error = RuntimeError(f"Unexpected health payload: {payload!r}")
        except Exception as exc:  # noqa: BLE001
            last_error = exc
        time.sleep(poll_interval_seconds)
    raise TimeoutError("Service did not become healthy") from last_error


def _build_multipart(
    field_name: str,
    filename: str,
    content: bytes,
    content_type: str,
) -> tuple[str, bytes]:
    boundary = f"----visionforge{uuid.uuid4().hex}"
    parts = [
        f"--{boundary}\r\n".encode("utf-8"),
        (
            f'Content-Disposition: form-data; name="{field_name}"; '
            f'filename="{filename}"\r\n'
        ).encode("utf-8"),
        f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"),
        content,
        b"\r\n",
        f"--{boundary}--\r\n".encode("utf-8"),
    ]
    return boundary, b"".join(parts)


def _predict(
    base_url: str,
    token: str,
    image_path: Path,
    profile: str,
    limit: int,
) -> dict[str, object]:
    predict_url = urllib.parse.urljoin(
        base_url,
        f"/predict?{urllib.parse.urlencode({'profile': profile, 'limit': limit})}",
    )
    boundary, body = _build_multipart(
        "file", image_path.name, image_path.read_bytes(), "image/jpeg"
    )
    request = urllib.request.Request(
        predict_url,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not payload.get("tags"):
        raise RuntimeError(f"Predict returned no tags: {payload!r}")
    return payload


def main() -> int:
    args = _parse_args()
    if not args.config_dir.is_dir():
        raise FileNotFoundError(f"Missing config directory: {args.config_dir}")
    if not args.data_dir.is_dir():
        raise FileNotFoundError(f"Missing data directory: {args.data_dir}")
    if not args.image_path.is_file():
        raise FileNotFoundError(f"Missing sample image: {args.image_path}")

    container_id = _start_container(args)
    try:
        health_payload = _wait_for_health(
            args.base_url, args.timeout_seconds, args.poll_interval_seconds
        )
        if health_payload.get("meta", {}).get("app_name") != "vision-forge-api":
            raise RuntimeError(f"Unexpected health response: {health_payload!r}")

        predict_payload = _predict(
            args.base_url,
            args.predict_token,
            args.image_path,
            args.profile,
            args.limit,
        )
        if predict_payload.get("meta", {}).get("profile") != args.profile:
            raise RuntimeError(f"Unexpected predict response: {predict_payload!r}")
        if len(predict_payload.get("tags", [])) > args.limit:
            raise RuntimeError(f"Predict returned too many tags: {predict_payload!r}")
    except Exception:  # noqa: BLE001
        _docker_logs(container_id)
        raise
    finally:
        _stop_container(container_id)

    print(json.dumps({"health": health_payload, "predict": predict_payload}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
