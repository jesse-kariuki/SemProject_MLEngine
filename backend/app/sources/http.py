"""Shared HTTP behaviour for external data sources."""
from __future__ import annotations

import time

import httpx

from app.config import get_settings


class SourceError(RuntimeError):
    """An external source could not provide usable data."""


def get_with_retries(client: httpx.Client, url: str, params: dict | None = None) -> httpx.Response:
    """GET with backoff on network errors, 429 and 5xx. Other 4xx mean our request is wrong, so fail at once."""
    settings = get_settings()
    problem = "no attempts made"
    for attempt in range(settings.http_retries):
        if attempt:
            time.sleep(2 ** (attempt - 1))
        try:
            response = client.get(url, params=params, timeout=settings.http_timeout_seconds, follow_redirects=True)
        except httpx.HTTPError as err:
            problem = repr(err)
            continue
        if response.status_code == 429 or response.status_code >= 500:
            problem = f"HTTP {response.status_code}"
            continue
        if response.status_code >= 400:
            raise SourceError(f"HTTP {response.status_code} from {url}: {response.text[:200]}")
        return response
    raise SourceError(f"{url} failed after {settings.http_retries} attempts: {problem}")
