"""CORS static file server for LiveView components.

Vitessce and agent-generated components read their data (and their own
code) over HTTP. That data lives in the workspace as local files, which a
browser cannot fetch by path. This module lazily starts a localhost,
CORS-enabled static HTTP server rooted at the workspace, so the LiveView
iframe can fetch workspace files by URL.

Localhost-bound — fine for the desktop app (browser and backend share a
machine). Supports HTTP range requests (needed for OME-TIFF / sharded Zarr).
"""

from __future__ import annotations

import socket
from pathlib import Path

from aiohttp import web

from pantheon.utils.log import logger


def _free_port() -> int:
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
    finally:
        s.close()


@web.middleware
async def _cors_middleware(request: web.Request, handler):
    """Allow the LiveView iframe (a different origin/port) to fetch."""
    if request.method == "OPTIONS":
        resp: web.StreamResponse = web.Response()
    else:
        resp = await handler(request)
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET, HEAD, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "*"
    return resp


class LiveViewDataServer:
    """Lazily-started localhost CORS static server for one workspace root.

    One server per backend process; rooted at the first workspace that asks.
    """

    def __init__(self):
        self._runner: web.AppRunner | None = None
        self._root: Path | None = None
        self._base_url: str | None = None

    async def ensure_started(self, root: Path) -> str:
        """Start the server (once) rooted at `root`; return its base URL."""
        if self._base_url is not None:
            return self._base_url

        root = Path(root).resolve()
        app = web.Application(middlewares=[_cors_middleware])
        app.router.add_static(
            "/", str(root), show_index=False, follow_symlinks=False,
        )
        runner = web.AppRunner(app)
        await runner.setup()
        port = _free_port()
        site = web.TCPSite(runner, "127.0.0.1", port)
        await site.start()

        self._runner = runner
        self._root = root
        self._base_url = f"http://127.0.0.1:{port}"
        logger.info(
            "live_view: data server started at {} (root={})", self._base_url, root,
        )
        return self._base_url

    @property
    def root(self) -> Path | None:
        return self._root

    @property
    def base_url(self) -> str | None:
        return self._base_url

    def url_for(self, abs_path: Path) -> str | None:
        """Map an absolute path under the server root to its served URL."""
        if self._root is None or self._base_url is None:
            return None
        try:
            rel = Path(abs_path).resolve().relative_to(self._root)
        except ValueError:
            return None  # outside the served root
        return f"{self._base_url}/{rel.as_posix()}"

    async def stop(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None
            self._base_url = None
            self._root = None
