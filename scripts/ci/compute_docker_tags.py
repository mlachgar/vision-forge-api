#!/usr/bin/env python3
"""Compute Docker tags for release and edge publication flows."""

from __future__ import annotations

import argparse
import os
import re
import sys

SEMVER_TAG_PATTERN = re.compile(r"^v(?P<version>\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?)$")


def _parse_release_version(ref: str, explicit_release_tag: str | None) -> str | None:
    candidate = explicit_release_tag
    if not candidate and ref.startswith("refs/tags/"):
        candidate = ref.removeprefix("refs/tags/")
    if not candidate:
        return None

    match = SEMVER_TAG_PATTERN.match(candidate)
    if not match:
        raise ValueError(
            f"Invalid release tag '{candidate}'. Expected format like 'v1.2.3' or 'v1.2.3-rc.1'."
        )
    return match.group("version")


def _compute_tags(
    *,
    image: str,
    variant: str,
    ref: str,
    event_name: str,
    latest_variant: str,
    release_tag: str | None,
    publish_edge_from_dispatch: bool,
) -> tuple[list[str], str | None]:
    version = _parse_release_version(ref, release_tag)
    if version:
        tags = [
            f"{image}:{version}-{variant}",
            f"{image}:{variant}",
        ]
        if variant == latest_variant:
            tags.append(f"{image}:latest")
        return tags, version

    if event_name == "push" and ref == "refs/heads/main":
        return [f"{image}:edge-{variant}"], None

    if event_name == "workflow_dispatch" and publish_edge_from_dispatch:
        return [f"{image}:edge-{variant}"], None

    return [], None


def _write_output(name: str, value: str) -> None:
    output_file = os.getenv("GITHUB_OUTPUT")
    if output_file:
        with open(output_file, "a", encoding="utf-8") as handle:
            handle.write(f"{name}={value}\n")
        return
    print(f"{name}={value}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute Docker tag list for the current build context.")
    parser.add_argument("--event-name", required=True)
    parser.add_argument("--ref", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--release-tag", default=None)
    parser.add_argument("--latest-variant", default="cpu-full")
    parser.add_argument("--publish-edge-from-dispatch", action="store_true")
    args = parser.parse_args()

    try:
        tags, version = _compute_tags(
            image=args.image,
            variant=args.variant,
            ref=args.ref,
            event_name=args.event_name,
            latest_variant=args.latest_variant,
            release_tag=args.release_tag,
            publish_edge_from_dispatch=args.publish_edge_from_dispatch,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    tags_output = ",".join(tags)
    primary_tag = tags[0] if tags else ""
    _write_output("publish", "true" if tags else "false")
    _write_output("tags", tags_output)
    _write_output("primary_tag", primary_tag)
    _write_output("version", version or "")
    _write_output("is_release", "true" if version else "false")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
