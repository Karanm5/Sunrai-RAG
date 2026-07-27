"""The environment check must be reliable on every platform.

Its whole purpose is catching setup problems in seconds rather than after a
long run fails, so it must never crash on a missing dependency or an offline
network -- the exact conditions it exists to report.
"""
import pytest

from sunrai_rag.config import Config


def test_runs_and_reports_every_section(capsys, monkeypatch):
    """Deliberately does not assert zero problems: the test must pass on any
    machine, including CI where the optional ML stack is not installed."""
    monkeypatch.setattr("sunrai_rag.ingest.ocr.find_tesseract_binary",
                        lambda p=None: "/usr/bin/tesseract")
    from sunrai_rag import cli

    cfg = Config()
    cfg.llm.backend = "stub"
    problems = cli.cmd_doctor(cfg)
    out = capsys.readouterr().out
    assert "ENVIRONMENT CHECK" in out
    assert "python" in out and "tesseract" in out and "GB" in out
    assert isinstance(problems, int) and problems >= 0


def test_reports_missing_tesseract_with_platform_help(capsys, monkeypatch):
    monkeypatch.setattr("sunrai_rag.ingest.ocr.find_tesseract_binary", lambda p=None: None)
    from sunrai_rag import cli

    cfg = Config()
    cfg.llm.backend = "stub"
    problems = cli.cmd_doctor(cfg)
    out = capsys.readouterr().out
    assert "tesseract not found" in out
    assert "Windows" in out and "macOS" in out and "Linux" in out
    assert problems >= 1


def test_reports_missing_api_key(capsys, monkeypatch):
    monkeypatch.setattr("sunrai_rag.ingest.ocr.find_tesseract_binary",
                        lambda p=None: "/usr/bin/tesseract")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    from sunrai_rag import cli

    cfg = Config()
    cfg.llm.backend = "groq"
    problems = cli.cmd_doctor(cfg)
    out = capsys.readouterr().out
    assert "GROQ_API_KEY is not set" in out
    assert problems >= 1


def test_local_and_stub_backends_need_no_key(capsys, monkeypatch):
    monkeypatch.setattr("sunrai_rag.ingest.ocr.find_tesseract_binary",
                        lambda p=None: "/usr/bin/tesseract")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    from sunrai_rag import cli

    for backend in ("stub", "local"):
        cfg = Config()
        cfg.llm.backend = backend
        cli.cmd_doctor(cfg)
        assert "needs no API key" in capsys.readouterr().out


def test_survives_network_failure_during_model_check(capsys, monkeypatch):
    """Offline must be reported, not raised -- doctor runs before anything works."""
    monkeypatch.setattr("sunrai_rag.ingest.ocr.find_tesseract_binary",
                        lambda p=None: "/usr/bin/tesseract")
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    import sys
    import types

    def exploding_get(*a, **k):
        raise ConnectionError("no network")

    monkeypatch.setitem(sys.modules, "requests",
                        types.SimpleNamespace(get=exploding_get, post=lambda *a, **k: None))
    from sunrai_rag import cli

    cfg = Config()
    cfg.llm.backend = "groq"
    cli.cmd_doctor(cfg)  # must not raise
    assert "could not verify model" in capsys.readouterr().out


def test_doctor_is_a_valid_cli_stage():
    from sunrai_rag import cli

    with pytest.raises(SystemExit):
        cli.main(["doctor", "--config", "configs/ci.yaml"])
