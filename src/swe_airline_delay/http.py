"""Tiny shared HTTP layer (retries, UA, streaming download).

Every fetcher goes through :func:`get` / :func:`download`, so tests can mock
this one module and no test ever touches the network.
"""

from __future__ import annotations

from pathlib import Path

import requests
from loguru import logger
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

USER_AGENT = "swe-airline-delay/0.1 (academic research pipeline)"
DEFAULT_TIMEOUT = 60

_session: requests.Session | None = None


def session() -> requests.Session:
    global _session
    if _session is None:
        s = requests.Session()
        retry = Retry(
            total=4,
            backoff_factor=2,
            status_forcelist=(500, 502, 503, 504),
            allowed_methods=("GET", "HEAD", "POST"),
        )
        s.mount("https://", HTTPAdapter(max_retries=retry))
        s.mount("http://", HTTPAdapter(max_retries=retry))
        s.headers["User-Agent"] = USER_AGENT
        _session = s
    return _session


def get(url: str, **kwargs) -> requests.Response:
    """GET with default timeout. Does NOT raise on HTTP errors (callers decide)."""
    kwargs.setdefault("timeout", DEFAULT_TIMEOUT)
    return session().get(url, **kwargs)


def post(url: str, **kwargs) -> requests.Response:
    kwargs.setdefault("timeout", DEFAULT_TIMEOUT)
    return session().post(url, **kwargs)


def download(url: str, dest: Path, chunk: int = 1 << 20) -> bool:
    """Stream ``url`` to ``dest`` atomically. Returns False on 404, raises on other errors."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    resp = get(url, stream=True, timeout=300)
    if resp.status_code == 404:
        logger.warning("404 Not Found: {}", url)
        return False
    resp.raise_for_status()
    with open(tmp, "wb") as fh:
        for block in resp.iter_content(chunk):
            fh.write(block)
    tmp.replace(dest)
    return True
