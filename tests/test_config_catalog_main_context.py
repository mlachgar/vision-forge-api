from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import vision_forge_api
from vision_forge_api import main as main_mod
from vision_forge_api.config import loader as loader_mod
from vision_forge_api.api import context_builder
from vision_forge_api.catalog.service import TagCatalog
from vision_forge_api.config.loader import ConfigLoader, _read_yaml
from vision_forge_api.config.schema import (
    AuthConfig,
    AuthRole,
    Profile,
    ProfilesConfig,
    PromptEntry,
    PromptsConfig,
    SettingsConfig,
    TagPrompt,
    TagSet,
    TagSetsConfig,
)


def _write_yaml(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def test_read_yaml_and_loader_roundtrip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_dir = tmp_path / "cfg"
    config_dir.mkdir()
    _write_yaml(config_dir / "auth.yaml", "token_prefix: vfk_\n")
    _write_yaml(
        config_dir / "settings.yaml",
        """
app_name: app
embeddings_dir: data/emb
model_cache_dir: data/model
siglip_model_id: google/siglip-base-patch16-224
""".strip(),
    )
    _write_yaml(
        config_dir / "tag_sets.yaml",
        "tag_sets:\n  - name: animals\n    canonical_tags: [cat, dog]\n",
    )
    _write_yaml(
        config_dir / "profiles.yaml",
        "profiles:\n  - name: default\n    tag_sets: [animals]\n",
    )
    _write_yaml(
        config_dir / "prompts.yaml",
        "prompts:\n  - canonical_tag: cat\n    prompts:\n      - template: 'a {tag}'\n",
    )

    monkeypatch.setenv("VISION_FORGE_CONFIG_DIR", str(config_dir))
    loader = ConfigLoader()

    assert loader.load_auth().token_prefix == "vfk_"
    assert loader.load_settings().app_name == "app"
    assert loader.load_tag_sets().tag_sets[0].name == "animals"
    assert loader.load_profiles().profiles[0].name == "default"
    assert loader.load_prompts().prompts[0].canonical_tag == "cat"


def test_read_yaml_validation_errors(tmp_path: Path) -> None:
    missing = tmp_path / "missing.yaml"
    with pytest.raises(FileNotFoundError):
        _read_yaml(missing)

    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("- not-a-mapping\n", encoding="utf-8")
    with pytest.raises(ValueError):
        _read_yaml(invalid)

    empty = tmp_path / "empty.yaml"
    empty.write_text("null\n", encoding="utf-8")
    assert _read_yaml(empty) == {}


def test_config_loader_uses_default_directory_when_env_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_dir = tmp_path / "default-config"
    config_dir.mkdir()
    _write_yaml(config_dir / "auth.yaml", "token_prefix: vfk_\n")
    _write_yaml(
        config_dir / "settings.yaml",
        """
app_name: app
embeddings_dir: data/emb
model_cache_dir: data/model
siglip_model_id: google/siglip-base-patch16-224
""".strip(),
    )
    _write_yaml(config_dir / "tag_sets.yaml", "tag_sets: []\n")
    _write_yaml(config_dir / "profiles.yaml", "profiles: []\n")
    _write_yaml(config_dir / "prompts.yaml", "prompts: []\n")

    monkeypatch.delenv("VISION_FORGE_CONFIG_DIR", raising=False)
    monkeypatch.setattr(loader_mod, "DEFAULT_CONFIG_DIR", config_dir)

    loader = ConfigLoader()

    assert loader.config_dir == config_dir
    assert loader.load_auth().token_prefix == "vfk_"


def test_config_loader_raises_for_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ConfigLoader(tmp_path / "missing")


def _catalog() -> TagCatalog:
    return TagCatalog(
        TagSetsConfig(
            tag_sets=(
                TagSet(name="animals", canonical_tags=("cat", "dog")),
                TagSet(name="city", canonical_tags=("street",)),
            )
        ),
        ProfilesConfig(
            profiles=(Profile(name="default", tag_sets=("animals", "city")),)
        ),
        PromptsConfig(
            prompts=(
                PromptEntry(
                    canonical_tag="cat", prompts=(TagPrompt(template="a {tag}"),)
                ),
            )
        ),
    )


def test_catalog_happy_path_and_accessors() -> None:
    catalog = _catalog()

    assert len(catalog.list_tag_sets()) == 2
    assert len(catalog.list_profiles()) == 1
    assert catalog.get_tag_set("animals").name == "animals"
    assert catalog.get_profile("default").name == "default"
    assert catalog.resolve_canonical_tags(("animals", "city")) == (
        "cat",
        "dog",
        "street",
    )
    detail = catalog.profile_detail("default")
    assert detail.canonical_tags == ("cat", "dog", "street")
    assert catalog.prompts_for_tag("cat")[0].template == "a {tag}"
    assert catalog.prompts_for_tag("dog") == ()
    assert catalog.canonical_tags() == ("cat", "dog", "street")


def test_catalog_validation_errors() -> None:
    with pytest.raises(ValueError):
        TagCatalog(
            TagSetsConfig(
                tag_sets=(
                    TagSet(name="dup", canonical_tags=("a",)),
                    TagSet(name="dup", canonical_tags=("b",)),
                )
            ),
            ProfilesConfig(profiles=(Profile(name="default", tag_sets=("dup",)),)),
            PromptsConfig(prompts=()),
        )

    with pytest.raises(ValueError):
        TagCatalog(
            TagSetsConfig(tag_sets=(TagSet(name="animals", canonical_tags=("cat",)),)),
            ProfilesConfig(
                profiles=(
                    Profile(name="default", tag_sets=("animals",)),
                    Profile(name="default", tag_sets=("animals",)),
                )
            ),
            PromptsConfig(prompts=()),
        )

    with pytest.raises(ValueError):
        TagCatalog(
            TagSetsConfig(tag_sets=(TagSet(name="animals", canonical_tags=("cat",)),)),
            ProfilesConfig(profiles=(Profile(name="default", tag_sets=("missing",)),)),
            PromptsConfig(prompts=()),
        )

    c = _catalog()
    with pytest.raises(KeyError):
        c.get_tag_set("missing")
    with pytest.raises(KeyError):
        c.get_profile("missing")


def test_context_builder_wires_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_loader = SimpleNamespace(
        config_dir=Path("/tmp/config"),
        load_settings=lambda: SettingsConfig(
            embeddings_dir=Path("/tmp/emb"), model_cache_dir=Path("/tmp/cache")
        ),
        load_tag_sets=lambda: TagSetsConfig(
            tag_sets=(TagSet(name="animals", canonical_tags=("cat",)),)
        ),
        load_profiles=lambda: ProfilesConfig(
            profiles=(Profile(name="default", tag_sets=("animals",)),)
        ),
        load_prompts=lambda: PromptsConfig(prompts=()),
        load_auth=lambda: AuthConfig(default_roles=(AuthRole.PREDICT,)),
    )

    monkeypatch.setenv("VISION_FORGE_DEVICE", "cpu")
    monkeypatch.setattr(context_builder, "TagCatalog", lambda *args: "catalog")
    monkeypatch.setattr(
        context_builder, "AuthTokenManager", lambda cfg: "token_manager"
    )
    monkeypatch.setattr(context_builder, "ApiKeyRepository", lambda: "repo")
    monkeypatch.setattr(
        context_builder,
        "AuthCache",
        SimpleNamespace(from_repository=lambda repo: "cache"),
    )
    monkeypatch.setattr(
        context_builder,
        "SiglipService",
        lambda model_id, cache_dir, device_hint=None: (
            "siglip",
            model_id,
            cache_dir,
            device_hint,
        ),
    )
    monkeypatch.setattr(
        context_builder,
        "PredictionService",
        lambda tag_catalog, siglip, embeddings_dir: (
            "predict",
            tag_catalog,
            siglip,
            embeddings_dir,
        ),
    )

    ctx = context_builder.build_context(fake_loader, "1.2.3")

    assert ctx.version == "1.2.3"
    assert ctx.tag_catalog == "catalog"
    assert ctx.auth_cache == "cache"
    assert ctx.token_manager == "token_manager"
    assert ctx.api_key_repo == "repo"
    assert ctx.siglip_service[0] == "siglip"
    assert ctx.prediction_service[0] == "predict"


def test_package_and_main_entrypoints(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(vision_forge_api, "pkg_version", lambda _: "9.9.9")
    assert vision_forge_api.resolve_version() == "9.9.9"

    class _PkgErr(Exception):
        pass

    monkeypatch.setattr(vision_forge_api, "PackageNotFoundError", _PkgErr)

    def _raise(_: str) -> str:
        raise _PkgErr()

    monkeypatch.setattr(vision_forge_api, "pkg_version", _raise)
    assert vision_forge_api.resolve_version() == vision_forge_api.CONFIG_VERSION

    called: dict[str, object] = {}

    def _fake_create_app(config_dir: str | None):
        called["config_dir"] = config_dir
        return "app"

    def _fake_run(app, host: str, port: int) -> None:
        called["app"] = app
        called["host"] = host
        called["port"] = port

    monkeypatch.setattr(main_mod, "create_app", _fake_create_app)
    monkeypatch.setattr(main_mod.uvicorn, "run", _fake_run)
    monkeypatch.setenv("VISION_FORGE_HOST", "127.0.0.1")
    monkeypatch.setenv("VISION_FORGE_PORT", "9090")
    monkeypatch.setenv("VISION_FORGE_CONFIG_DIR", "/tmp/config")

    main_mod.main()

    assert called == {
        "config_dir": "/tmp/config",
        "app": "app",
        "host": "127.0.0.1",
        "port": 9090,
    }


def test_package_create_app_delegates_to_api_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called: dict[str, object] = {}

    monkeypatch.setattr(
        "vision_forge_api.api.app.create_app",
        lambda config_dir=None: called.update({"config_dir": config_dir}) or "app",
    )

    app = vision_forge_api.create_app("/tmp/config")

    assert app == "app"
    assert called["config_dir"] == "/tmp/config"


def test_main_uses_default_environment_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called: dict[str, object] = {}

    monkeypatch.delenv("VISION_FORGE_HOST", raising=False)
    monkeypatch.delenv("VISION_FORGE_PORT", raising=False)
    monkeypatch.delenv("VISION_FORGE_CONFIG_DIR", raising=False)
    monkeypatch.setattr(main_mod, "create_app", lambda config_dir: "app")
    monkeypatch.setattr(
        main_mod.uvicorn,
        "run",
        lambda app, host, port: called.update({"app": app, "host": host, "port": port}),
    )

    main_mod.main()

    assert called == {"app": "app", "host": "0.0.0.0", "port": 8000}


def test_main_module_guard_executes_main(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called: dict[str, object] = {}

    monkeypatch.setattr(
        "vision_forge_api.api.app.create_app", lambda config_dir=None: "app"
    )
    monkeypatch.setattr(
        main_mod.uvicorn,
        "run",
        lambda app, host, port: called.update({"app": app, "host": host, "port": port}),
    )

    import runpy

    runpy.run_module("vision_forge_api.main", run_name="__main__")

    assert called == {"app": "app", "host": "0.0.0.0", "port": 8000}
