"""LiveViewToolSet — open and control agent-driven UI components.

A "LiveView" is a UI component the agent can open, drive, and observe. The
authoritative state lives here in a per-view ``LiveViewSession``. Two kinds of
client mutate / read it:

  * the **agent**, via the ``@tool`` methods (open / update / set_state /
    call / get_state) — these also broadcast the change to the UI;
  * the **UI**, via the ``@tool(exclude=True)`` methods (report_view_state /
    report_action_result) — when the user interacts with the component, or
    an agent-invoked action finishes, the UI reports back so the agent's
    next get_state / the pending call sees it.

Transport: state-change events publish on the existing NATS chat stream
(``pantheon.stream.chat_<chat_id>``) with ``live_view.*`` event types. The
component side speaks the matching bridge protocol via live-view-sdk.js.

This toolset is generic; ``vitessce`` is simply the first registered
``view_type``. The component's "state" is opaque here — for Vitessce it is
the Vitessce view config; a patch deep-merges into it.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from pantheon.toolset import ToolSet, tool
from pantheon.utils.log import logger

# Component types the UI knows how to render.
#   vitessce : the Vitessce data-browser adapter
#   custom   : an agent-generated component — state must carry `module_url`,
#              the served URL of a JS module exporting setup(lv, root)
KNOWN_VIEW_TYPES = {"vitessce", "custom"}

# How long live_view_call waits for the component to return an action result.
ACTION_TIMEOUT_SECONDS = 30


def _deep_merge(target: Any, patch: Any) -> Any:
    """Deep-merge ``patch`` into ``target``, returning a new value."""
    if not isinstance(patch, dict):
        return patch
    base = target if isinstance(target, dict) else {}
    out = dict(base)
    for key, value in patch.items():
        out[key] = _deep_merge(base.get(key), value)
    return out


@dataclass
class LiveViewSession:
    """Authoritative state for one open LiveView."""

    view_id: str
    view_type: str
    title: str
    chat_id: str
    state: dict[str, Any] = field(default_factory=dict)
    status: str = "opening"  # opening | ready | error | closed
    error: str | None = None
    updated_at: float = field(default_factory=time.time)

    def snapshot(self) -> dict[str, Any]:
        """Structured state the agent reads to decide its next action."""
        return {
            "view_id": self.view_id,
            "view_type": self.view_type,
            "title": self.title,
            "status": self.status,
            "state": self.state,
            "error": self.error,
            "updated_at": self.updated_at,
        }


class LiveViewToolSet(ToolSet):
    """Toolset for opening and controlling agent-driven UI components."""

    def __init__(self, name: str = "live_view", **kwargs):
        super().__init__(name, **kwargs)
        self._views: dict[str, LiveViewSession] = {}
        # action_id -> Future, resolved by report_action_result.
        self._pending_actions: dict[str, asyncio.Future] = {}
        self._nats = None  # lazy NATSStreamAdapter
        self._data_server = None  # lazy LiveViewDataServer

    # ── internals ─────────────────────────────────────────────────────────

    def _chat_id(self) -> str | None:
        """Resolve the chat id from the execution context.

        Agent tool calls carry it as `chat_id` (injected by room.chat); the
        UI's proxy_toolset path injects it as `session_id`. NOT `client_id`
        (that is the UI connection id, stable across chats).
        """
        ctx = self.get_context() or {}
        return ctx.get("session_id") or ctx.get("chat_id")

    async def _publish(self, chat_id: str, event: dict[str, Any]) -> None:
        """Broadcast a live_view.* event to the UI over the NATS chat stream."""
        if not chat_id:
            logger.warning("live_view: no chat_id, cannot publish {}", event.get("type"))
            return
        if self._nats is None:
            from pantheon.chatroom.stream import NATSStreamAdapter

            self._nats = NATSStreamAdapter()
        try:
            await self._nats.publish(chat_id, event["type"], event)
        except Exception as e:  # streaming is best-effort
            logger.error("live_view: publish failed: {}", e)

    def _require(self, view_id: str) -> LiveViewSession:
        session = self._views.get(view_id)
        if session is None:
            raise KeyError(f"No LiveView with id '{view_id}'")
        return session

    # ── agent-facing tools ────────────────────────────────────────────────

    @tool
    async def open_live_view(self, view_type: str, title: str, state: dict) -> dict:
        """Open an interactive visualization component in the Pantheon UI sidebar.

        The component renders in the right sidebar; drive and observe it
        afterwards with the other live_view tools.

        Args:
            view_type: Component type:
                - "vitessce": the Vitessce spatial / single-cell / imaging
                  data browser.
                - "custom": an agent-generated component. `state` MUST include
                  `module_url` — the URL (from serve_local_data) of a JS
                  module exporting `setup(lv, root)`.
            title: Short title shown on the sidebar tab.
            state: The component's initial state. For "vitessce" this is a
                Vitessce *view config* JSON object (keys: version, name,
                datasets, coordinationSpace, layout, initStrategy); data file
                URLs must be browser-reachable (public, or served via
                serve_local_data). For "custom" it is the component's own
                initial state, plus the required `module_url`.

        Returns:
            dict with success, view_id (use it for the other tools), and the
            initial state snapshot.
        """
        if view_type not in KNOWN_VIEW_TYPES:
            return {
                "success": False,
                "error": f"Unknown view_type '{view_type}'. Known: {sorted(KNOWN_VIEW_TYPES)}",
            }
        if view_type == "custom" and not (state or {}).get("module_url"):
            return {
                "success": False,
                "error": (
                    "view_type 'custom' requires state.module_url — serve "
                    "your component module with serve_local_data first."
                ),
            }
        chat_id = self._chat_id()
        if not chat_id:
            return {"success": False, "error": "No active chat context"}

        view_id = f"lv-{uuid.uuid4().hex[:12]}"
        session = LiveViewSession(
            view_id=view_id,
            view_type=view_type,
            title=title,
            chat_id=chat_id,
            state=state or {},
        )
        self._views[view_id] = session

        await self._publish(chat_id, {
            "type": "live_view.open",
            "view_id": view_id,
            "view_type": view_type,
            "title": title,
            "state": session.state,
        })
        logger.info("live_view: opened {} ({}) for chat {}", view_id, view_type, chat_id)
        return {"success": True, "view_id": view_id, "state": session.snapshot()}

    @tool
    async def live_view_update(self, view_id: str, patch: dict) -> dict:
        """Apply a partial-state patch to an open LiveView (drive the component).

        The patch is deep-merged into the component's current state. This is
        the main way to drive a view incrementally.

        For "vitessce", the state is the view config and coordination values
        live under `coordinationSpace`. Example — zoom a spatial view:
            patch = {"coordinationSpace": {"spatialZoom": {"A": 4}}}
        Other Vitessce coordination types: spatialTargetX / spatialTargetY,
        obsColorEncoding, featureSelection, obsSetSelection, obsHighlight.

        Args:
            view_id: id returned by open_live_view.
            patch: a partial state object to deep-merge into the current state.

        Returns:
            dict with success and the resulting state snapshot.
        """
        try:
            session = self._require(view_id)
        except KeyError as e:
            return {"success": False, "error": str(e)}

        session.state = _deep_merge(session.state, patch or {})
        session.updated_at = time.time()
        await self._publish(session.chat_id, {
            "type": "live_view.patch",
            "view_id": view_id,
            "patch": patch or {},
        })
        return {"success": True, "state": session.snapshot()}

    @tool
    async def live_view_set_state(self, view_id: str, state: dict) -> dict:
        """Replace an open LiveView's whole state.

        Use for structural changes (different dataset / layout). Prefer
        live_view_update for incremental changes.

        Args:
            view_id: id returned by open_live_view.
            state: the new full component state.

        Returns:
            dict with success and the resulting state snapshot.
        """
        try:
            session = self._require(view_id)
        except KeyError as e:
            return {"success": False, "error": str(e)}

        session.state = state or {}
        session.updated_at = time.time()
        await self._publish(session.chat_id, {
            "type": "live_view.set",
            "view_id": view_id,
            "state": session.state,
        })
        return {"success": True, "state": session.snapshot()}

    @tool
    async def live_view_call(self, view_id: str, action: str, args: dict = {}) -> dict:
        """Invoke a named action the component exposes, and wait for its result.

        Components register actions via the LiveView SDK's defineAction(). Not
        all view types expose actions — "vitessce" is driven through
        live_view_update; agent-generated components typically expose actions.

        Args:
            view_id: id returned by open_live_view.
            action: the action name.
            args: arguments passed to the action handler.

        Returns:
            dict with success and `result` (the action's return value), or an
            error if the action failed / timed out.
        """
        try:
            session = self._require(view_id)
        except KeyError as e:
            return {"success": False, "error": str(e)}

        action_id = uuid.uuid4().hex
        loop = asyncio.get_event_loop()
        future: asyncio.Future = loop.create_future()
        self._pending_actions[action_id] = future

        await self._publish(session.chat_id, {
            "type": "live_view.action",
            "view_id": view_id,
            "action_id": action_id,
            "name": action,
            "args": args or {},
        })
        try:
            result = await asyncio.wait_for(future, timeout=ACTION_TIMEOUT_SECONDS)
            return {"success": True, "result": result}
        except asyncio.TimeoutError:
            return {"success": False, "error": f"action '{action}' timed out"}
        except Exception as e:
            return {"success": False, "error": str(e)}
        finally:
            self._pending_actions.pop(action_id, None)

    @tool
    async def live_view_get_state(self, view_id: str) -> dict:
        """Read an open LiveView's current state.

        Returns the latest state — including changes the *user* made by
        interacting with the component directly. Call this before deciding
        how to drive the view further.

        Args:
            view_id: id returned by open_live_view.
        """
        try:
            session = self._require(view_id)
        except KeyError as e:
            return {"success": False, "error": str(e)}
        return {"success": True, "state": session.snapshot()}

    @tool
    async def serve_local_data(self, path: str) -> dict:
        """Expose a local workspace file or directory over HTTP (CORS).

        LiveView components run in the browser and fetch their data — and, for
        agent-generated components, their own code — over HTTP. Local
        workspace paths are not browser-fetchable; this lazily starts a
        localhost CORS static server and returns a URL for `path`.

        Use this to make data servable before referencing it from a Vitessce
        view config, or to serve an agent-written component module before
        opening it with open_live_view(view_type="custom").

        Args:
            path: Absolute path, or path relative to the workspace, to a file
                or directory to serve.

        Returns:
            dict with success, base_url, and url (the URL for `path`).
        """
        from pathlib import Path

        p = Path(path)
        if not p.is_absolute():
            from pantheon.settings import get_settings
            p = get_settings().work_dir / p
        p = p.resolve()
        if not p.exists():
            return {"success": False, "error": f"Path does not exist: {p}"}

        if self._data_server is None:
            from .data_server import LiveViewDataServer
            self._data_server = LiveViewDataServer()

        # The first call fixes the served root (the dir of the path); later
        # calls must reference something under it.
        if self._data_server.root is None:
            await self._data_server.ensure_started(p if p.is_dir() else p.parent)

        url = self._data_server.url_for(p)
        if url is None:
            return {
                "success": False,
                "error": (
                    f"Path is outside the data server root "
                    f"'{self._data_server.root}'. Put data to serve under "
                    f"that directory."
                ),
            }
        return {
            "success": True,
            "base_url": self._data_server.base_url,
            "url": url,
        }

    @tool
    async def list_live_views(self) -> dict:
        """List the LiveViews currently open in this chat."""
        chat_id = self._chat_id()
        views = [
            s.snapshot() for s in self._views.values()
            if s.chat_id == chat_id and s.status != "closed"
        ]
        return {"success": True, "views": views}

    @tool
    async def close_live_view(self, view_id: str) -> dict:
        """Close an open LiveView and remove its sidebar tab.

        Args:
            view_id: id returned by open_live_view.
        """
        session = self._views.get(view_id)
        if session is None:
            return {"success": False, "error": f"No LiveView with id '{view_id}'"}
        session.status = "closed"
        await self._publish(session.chat_id, {
            "type": "live_view.close",
            "view_id": view_id,
        })
        return {"success": True}

    # ── UI-facing methods (not exposed to the LLM) ────────────────────────

    @tool(exclude=True)
    async def report_view_state(
        self,
        view_id: str,
        state: dict | None = None,
        status: str | None = None,
        error: str | None = None,
    ) -> dict:
        """UI → backend: report the component's state after a user interaction
        (or a lifecycle change), keeping the authoritative state in sync."""
        session = self._views.get(view_id)
        if session is None:
            return {"success": False, "error": f"No LiveView with id '{view_id}'"}
        if state is not None:
            session.state = state
        if status is not None:
            session.status = status
        if error is not None:
            session.error = error
        session.updated_at = time.time()
        return {"success": True}

    @tool(exclude=True)
    async def report_action_result(
        self,
        view_id: str,
        action_id: str,
        ok: bool,
        value: Any = None,
        error: str | None = None,
    ) -> dict:
        """UI → backend: deliver the result of an agent-invoked action,
        resolving the pending live_view_call."""
        future = self._pending_actions.get(action_id)
        if future is not None and not future.done():
            if ok:
                future.set_result(value)
            else:
                future.set_exception(RuntimeError(error or "action failed"))
        return {"success": True}

    @tool(exclude=True)
    async def list_views_for_ui(self, chat_id: str | None = None) -> dict:
        """UI → backend: fetch open views (e.g. to restore them after reload)."""
        cid = chat_id or self._chat_id()
        views = [
            s.snapshot() for s in self._views.values()
            if s.chat_id == cid and s.status != "closed"
        ]
        return {"success": True, "views": views}
