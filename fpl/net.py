"""Shared polite-HTTP cache helper for the external data sources in
``fpl/elo.py``, ``fpl/understat.py``, and ``fpl/odds.py``.

Mirrors the caching/retry/backoff philosophy of ``fpl/fetch.py``'s ``get()``,
but is reused across modules that talk to different hosts with different
response bodies (HTML, JSON, ...), so it works on raw response text instead
of parsed JSON.
"""

import time
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import requests


def redact(url: str) -> str:
    """Strip the query string from `url` so secrets (e.g. an API key) never
    end up in a log line or an exception message."""
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def cached_get(
    session: requests.Session,
    url: str,
    cache_file: Path,
    *,
    timeout: int = 30,
    retries: int = 3,
    pause: float = 0.2,
) -> str | None:
    """GET `url`, caching the raw response text at `cache_file`.

    A cache hit never touches the network. A 404 is cached as a permanent
    "not found" and returns ``None`` without retrying. Any other 4xx (bad
    API key, rate limit, bad request) is *not* cached, so a later run can
    retry fresh once the problem is fixed, and returns ``None`` after
    printing a short, secret-free diagnostic. 5xx and connection errors are
    retried up to `retries` times with exponential backoff and a `pause`
    pause follows every real request.
    """
    if cache_file.exists():
        text = cache_file.read_text()
        return text or None

    last_error: requests.RequestException | None = None
    for attempt in range(retries):
        try:
            r = session.get(url, timeout=timeout)
            if r.status_code == 404:
                cache_file.write_text("")
                time.sleep(pause)
                return None
            if 400 <= r.status_code < 500:
                print(f"  {redact(url)} -> HTTP {r.status_code}: {r.text[:200]!r}")
                time.sleep(pause)
                return None
            r.raise_for_status()
            cache_file.write_text(r.text)
            time.sleep(pause)
            return r.text
        except requests.RequestException as exc:
            last_error = exc
            time.sleep(2**attempt)
    raise RuntimeError(f"failed: {redact(url)}") from last_error
