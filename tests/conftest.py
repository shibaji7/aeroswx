import matplotlib
import pytest

matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def tmp_data(tmp_path, monkeypatch):
    """Every test writes into its own throw-away data dir; nothing touches ./data."""
    monkeypatch.setenv("SWE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("OPENSKY_CLIENT_ID", raising=False)
    monkeypatch.delenv("OPENSKY_CLIENT_SECRET", raising=False)
    return tmp_path / "data"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Fail loudly if any code path tries a real HTTP request."""

    def boom(*a, **k):
        raise AssertionError("real network access attempted in a test")

    import requests

    monkeypatch.setattr(requests.Session, "request", boom)


class FakeResponse:
    def __init__(self, status=200, json_data=None, text="", content=b"", headers=None):
        self.status_code = status
        self._json = json_data
        self.text = text
        self.content = content
        self.headers = headers or {}

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def iter_content(self, n):
        yield self.content


@pytest.fixture(autouse=True)
def clean_loguru():
    """CLI tests add sinks bound to captured stderr; drop them after each test."""
    from loguru import logger

    yield
    logger.remove()
