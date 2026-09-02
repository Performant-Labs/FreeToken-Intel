"""Daemon client used by `ft ctl`.

Upstream NVIDIA path: python/freetoken/daemon/client.py
Issue: `shell-daemon` (#27, see docs/architecture.md).

A plain-``urllib`` HTTP client for the daemon's control-plane app
(:mod:`freetoken.daemon.app`) -- no new dependency (``requests``/``httpx``)
just to talk to a local control socket, matching the project's existing
"dependency-light" client-side philosophy (``server/openai_api.py``'s own
docstring). This is also what a client-side ``ft ctl`` runs *without* an XPU
in its own process -- it only ever does JSON-over-HTTP.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request

DEFAULT_BASE_URL = "http://127.0.0.1:8500"


class DaemonConnectionError(ConnectionError):
    """The daemon at ``base_url`` could not be reached (not running, wrong
    port, ...). Distinct from an HTTP error response (e.g. 409), which
    means the daemon *is* reachable but rejected the request."""


class DaemonClient:
    def __init__(self, base_url: str = DEFAULT_BASE_URL) -> None:
        self.base_url = base_url.rstrip("/")

    def _request(self, method: str, path: str, body: dict | None = None) -> dict:
        data = json.dumps(body).encode("utf-8") if body is not None else None
        headers = {"Content-Type": "application/json"} if data is not None else {}
        req = urllib.request.Request(f"{self.base_url}{path}", data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"daemon returned {exc.code} for {method} {path}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise DaemonConnectionError(f"could not reach daemon at {self.base_url}: {exc.reason}") from exc

    def status(self) -> dict:
        return self._request("GET", "/status")

    def start(self, model: str, *, host: str = "127.0.0.1", port: int = 8080, extra_args: list[str] | None = None) -> dict:
        return self._request(
            "POST",
            "/start",
            {"model": model, "host": host, "port": port, "extra_args": list(extra_args) if extra_args else []},
        )

    def stop(self) -> dict:
        return self._request("POST", "/stop", {})
