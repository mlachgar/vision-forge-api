"""Tag catalog and profile helpers."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable, Mapping

from ..config.schema import (
    Profile,
    ProfilesConfig,
    PromptsConfig,
    TagPrompt,
    TagSet,
    TagSetsConfig,
)


@dataclass(frozen=True)
class ProfileDetail:
    profile: Profile
    tag_sets: tuple[TagSet, ...]
    canonical_tags: tuple[str, ...]


class TagCatalog:
    """In-memory representation of tag sets, profiles, and prompts."""

    def __init__(
        self,
        tag_sets_config: TagSetsConfig,
        profiles_config: ProfilesConfig,
        prompts_config: PromptsConfig,
    ) -> None:
        self._tag_sets: Mapping[str, TagSet] = {
            ts.name: ts for ts in tag_sets_config.tag_sets
        }
        if len(self._tag_sets) != len(tag_sets_config.tag_sets):
            raise ValueError("Duplicate tag set names are not allowed")
        self._profiles: Mapping[str, Profile] = {
            profile.name: profile for profile in profiles_config.profiles
        }
        if len(self._profiles) != len(profiles_config.profiles):
            raise ValueError("Duplicate profile names are not allowed")
        tags: OrderedDict[str, None] = OrderedDict()
        for tag_set in tag_sets_config.tag_sets:
            for canonical in tag_set.canonical_tags:
                tags.setdefault(canonical, None)
        self._prompts: Mapping[str, tuple[TagPrompt, ...]] = {
            entry.canonical_tag: tuple(entry.prompts)
            for entry in prompts_config.prompts
        }
        self._validate_profile_tag_sets()
        self._canonical_keys = tuple(tags.keys())

    def _validate_profile_tag_sets(self) -> None:
        for profile in self._profiles.values():
            for tag_set_name in profile.tag_sets:
                if tag_set_name not in self._tag_sets:
                    raise ValueError(
                        f"Profile '{profile.name}' references unknown tag set '{tag_set_name}'"
                    )

    def list_tag_sets(self) -> tuple[TagSet, ...]:
        return tuple(self._tag_sets.values())

    def get_tag_set(self, name: str) -> TagSet:
        try:
            return self._tag_sets[name]
        except KeyError as exc:
            raise KeyError(f"Tag set '{name}' not defined") from exc

    def list_profiles(self) -> tuple[Profile, ...]:
        return tuple(self._profiles.values())

    def get_profile(self, name: str) -> Profile:
        try:
            return self._profiles[name]
        except KeyError as exc:
            raise KeyError(f"Profile '{name}' not defined") from exc

    def resolve_canonical_tags(self, tag_set_names: Iterable[str]) -> tuple[str, ...]:
        seen: OrderedDict[str, None] = OrderedDict()
        for tag_set_name in tag_set_names:
            tag_set = self.get_tag_set(tag_set_name)
            for tag_key in tag_set.canonical_tags:
                seen.setdefault(tag_key, None)
        return tuple(seen.keys())

    def profile_detail(self, profile_name: str) -> ProfileDetail:
        profile = self.get_profile(profile_name)
        tag_sets = tuple(self.get_tag_set(name) for name in profile.tag_sets)
        canonical_tags = self.resolve_canonical_tags(profile.tag_sets)
        return ProfileDetail(
            profile=profile, tag_sets=tag_sets, canonical_tags=canonical_tags
        )

    def prompts_for_tag(self, canonical_tag: str) -> tuple[TagPrompt, ...]:
        return self._prompts.get(canonical_tag, ())

    def canonical_tags(self) -> tuple[str, ...]:
        return self._canonical_keys
