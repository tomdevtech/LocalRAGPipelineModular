"""Tests for the Settings dataclass (defaults, YAML round trip, validation)."""
import pytest

from config.settings import Settings


@pytest.fixture
def paths(tmp_path):
    """Redirect all file-system paths into the temporary directory."""
    return {"data_path": str(tmp_path), "vector_db_path": str(tmp_path / "db")}


def test_default_values(paths):
    """Default values match the documented configuration."""
    settings = Settings(**paths)
    assert settings.k == 5
    assert settings.chunking_strategy == "recursive"
    assert settings.use_reranking is True
    assert isinstance(settings.data_path, str)


def test_from_yaml_success(tmp_path, paths):
    """Values from a YAML file override the defaults."""
    config = tmp_path / "config.yaml"
    config.write_text(
        'collection_name: "test_collection"\nchunk_size: 500\nuse_reranking: false\n'
        f'data_path: "{paths["data_path"]}"\nvector_db_path: "{paths["vector_db_path"]}"\n'
    )
    settings = Settings.from_yaml(str(config))
    assert settings.collection_name == "test_collection"
    assert settings.chunk_size == 500
    assert settings.use_reranking is False


def test_yaml_roundtrip(tmp_path, paths):
    """Saving to YAML and loading it again reproduces the same Settings."""
    original = Settings(chunk_size=300, chunk_overlap=30, **paths)
    target = tmp_path / "out.yaml"
    original.to_yaml(str(target))
    assert Settings.from_yaml(str(target)) == original


def test_from_yaml_file_not_found():
    """A missing YAML file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        Settings.from_yaml("non_existent_file.yaml")


def test_from_yaml_without_pyyaml(monkeypatch):
    """Without PyYAML installed, loading raises a helpful ImportError."""
    monkeypatch.setattr("config.settings.yaml", None)
    with pytest.raises(ImportError, match="PyYAML is not installed"):
        Settings.from_yaml("dummy_path.yaml")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"chunk_size": 0},
        {"chunk_size": 100, "chunk_overlap": 100},
        {"chunk_overlap": -1},
        {"k": 0},
        {"rerank_top_k": 0},
    ],
)
def test_invalid_values_are_rejected(paths, kwargs):
    """Nonsensical sizes and counts are rejected at construction time."""
    with pytest.raises(ValueError):
        Settings(**paths, **kwargs)
