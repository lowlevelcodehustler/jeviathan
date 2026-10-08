"""Control-character validation for the model-dir resolvers (nightly backlog).

Both entry points resolve the weights dir from --model-dir > $JEVIATHAN_MODEL_DIR
> .jeviathan_model_dir; a corrupted value must fail fast with a clear message
instead of surfacing as an opaque downstream error. No GPU required.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import pytest  # noqa: E402

import manage  # noqa: E402
import transformers_server  # noqa: E402


def test_manage_rejects_control_chars_in_cli_value():
    with pytest.raises(SystemExit, match="control characters"):
        manage.resolve_shim_model_dir("C:\\weights\\model\x00run")


def test_manage_rejects_control_chars_in_env(monkeypatch):
    monkeypatch.setenv("JEVIATHAN_MODEL_DIR", "E:\\weights\\bad\x01dir")
    with pytest.raises(SystemExit, match="control characters"):
        manage.resolve_shim_model_dir(None)


def test_transformers_server_rejects_newline_in_cli_value():
    with pytest.raises(SystemExit, match="control characters"):
        transformers_server.resolve_model_dir("a\nb")


def test_clean_cli_value_passes_through(monkeypatch):
    monkeypatch.delenv("JEVIATHAN_MODEL_DIR", raising=False)
    assert manage.resolve_shim_model_dir("/some/clean/dir") == "/some/clean/dir"
    assert transformers_server.resolve_model_dir("/some/clean/dir") == "/some/clean/dir"


def test_manage_unset_raises_clear_error(monkeypatch, tmp_path):
    monkeypatch.delenv("JEVIATHAN_MODEL_DIR", raising=False)
    monkeypatch.setattr(manage, "REPO", tmp_path)  # no .jeviathan_model_dir there
    with pytest.raises(SystemExit, match="No model dir"):
        manage.resolve_shim_model_dir(None)
