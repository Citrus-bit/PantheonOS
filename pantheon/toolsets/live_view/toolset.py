"""LiveViewToolSet — open and control agent-driven UI components.

Design
------
A "LiveView" is a UI component the agent can open, drive, and observe. The
authoritative state lives here, in a per-view ``LiveViewSession``. Two kinds
of client mutate / read it:

  * the **agent**, via the ``@tool`` methods (open / set_coordination /
    set_config / get_state) — these also broadcast the change to the UI;
  * the **UI**, via the ``@tool(exclude=True)`` methods (report_view_state /
    list_views) — when the user interacts with the component, the UI reports
    the resulting state back so the agent's next ``get_state`` sees it.

Transport: state-change events are published on the existing NATS chat
stream (``pantheon.stream.chat_<chat_id>``) with ``live_view.*`` event types,
so the UI needs no new subscription plumbing. UI → backend calls use the
standard ``proxy_toolset`` RPC.

This toolset is generic; ``vitessce`` is simply the first registered
``view_type``. Adding another (a genome browser, a plotly dashboard, …)
needs no change here — only a new frontend component in the UI registry.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from pantheon.toolset import ToolSet, tool
from pantheon.utils.log import logger

# Component types the UI knows how to render. Kept as a guard so the agent
# gets a clear error instead of opening a view the UI will not mount.
KNOWN_VIEW_TYPES = {"vitessce"}


@dataclass
class LiveViewSession:
    """Authoritative state for one open LiveView."""

    view_id: str
    view_type: str
    title: str
    chat_id: str
    config: dict[str, Any] = field(default_factory=dict)
    coordination_space: dict[str, Any] = field(default_factory=dict)
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
            "config": self.config,
            "coordination_space": self.coordination_space,
            "error": self.error,
            "updated_at": self.updated_at,
        }


class LiveViewToolSet(ToolSet):
    """Toolset for opening and controlling agent-driven UI components."""

    def __init__(self, name: str = "live_view", **kwargs):
        super().__init__(name, **kwargs)
        # view_id -> session. One toolset instance may serve several chats;
        # sessions carry their own chat_id and are filtered on read.
        self._views: dict[str, LiveViewSession] = {}
        self._nats = None  # lazy NATSStreamAdapter

    # ── internals ─────────────────────────────────────────────────────────

    def _chat_id(self) -> str | None:
        """Resolve the current chat id from the execution context.

        The agent's tool context carries it as `chat_id` (injected by
        room.chat); the UI's proxy_toolset path injects it as `session_id`.
        NOT `client_id` — that is the UI connection id, stable across chats,
        and publishing on it would target the wrong NATS subject.
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
    async def open_live_view(
        self,
        view_type: str,
        title: str,
        config: dict,
    ) -> dict:
        """Open an interactive visualization component in the Pantheon UI sidebar.

        The component is rendered in the right sidebar and you can drive and
        observe it afterwards with the other live_view tools.

        Args:
            view_type: Component type. Currently supported: "vitessce" (a
                spatial / single-cell / imaging data browser).
            title: Short human-readable title shown on the sidebar tab.
            config: The component's initial configuration. For "vitessce"
                this is a Vitessce *view config* JSON object (keys: version,
                name, datasets, coordinationSpace, layout, initStrategy).
                The `datasets[].files[].url` must point at data reachable by
                the browser (a public dataset, or one served with CORS).

        Returns:
            dict with success, view_id (use it for subsequent calls), and the
            initial state snapshot.
        """
        if view_type not in KNOWN_VIEW_TYPES:
            return {
                "success": False,
                "error": f"Unknown view_type '{view_type}'. Known: {sorted(KNOWN_VIEW_TYPES)}",
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
            config=config or {},
            coordination_space=(config or {}).get("coordinationSpace", {}) or {},
        )
        self._views[view_id] = session

        await self._publish(chat_id, {
            "type": "live_view.open",
            "view_id": view_id,
            "view_type": view_type,
            "title": title,
            "config": session.config,
        })
        logger.info("live_view: opened {} ({}) for chat {}", view_id, view_type, chat_id)
        return {"success": True, "view_id": view_id, "state": session.snapshot()}

    @tool
    async def live_view_set_coordination(self, view_id: str, updates: list[dict]) -> dict:
        """Change coordination values in an open LiveView (drive the component).

        For Vitessce, coordination values ARE the view state: zoom, pan,
        cell-set selection, color encoding, selected feature, etc. Each view
        is linked to the coordination space, so changing a value updates
        every linked panel.

        Args:
            view_id: id returned by open_live_view.
            updates: list of patches, each a dict with:
                - coordinationType (str): e.g. "spatialZoom", "spatialTargetX",
                  "spatialTargetY", "obsColorEncoding", "featureSelection",
                  "obsSetSelection", "obsHighlight".
                - coordinationScope (str, optional): scope name, default "A".
                - value: the new value for that coordination type.

        Returns:
            dict with success and the resulting state snapshot.
        """
        try:
            session = self._require(view_id)
        except KeyError as e:
            return {"success": False, "error": str(e)}

        for u in updates or []:
            ctype = u.get("coordinationType")
            if not ctype:
                continue
            scope = u.get("coordinationScope", "A")
            session.coordination_space.setdefault(ctype, {})[scope] = u.get("value")
        # keep config.coordinationSpace consistent
        session.config.setdefault("coordinationSpace", {})
        session.config["coordinationSpace"] = session.coordination_space
        session.updated_at = time.time()

        await self._publish(session.chat_id, {
            "type": "live_view.state_delta",
            "view_id": view_id,
            "updates": updates or [],
        })
        return {"success": True, "state": session.snapshot()}

    @tool
    async def live_view_set_config(self, view_id: str, config: dict) -> dict:
        """Replace an open LiveView's whole configuration.

        Use this for structural changes (different dataset, different panel
        layout). For incremental state changes prefer live_view_set_coordination.

        Args:
            view_id: id returned by open_live_view.
            config: the new full component config.

        Returns:
            dict with success and the resulting state snapshot.
        """
        try:
            session = self._require(view_id)
        except KeyError as e:
            return {"success": False, "error": str(e)}

        session.config = config or {}
        session.coordination_space = session.config.get("coordinationSpace", {}) or {}
        session.updated_at = time.time()

        await self._publish(session.chat_id, {
            "type": "live_view.set_config",
            "view_id": view_id,
            "config": session.config,
        })
        return {"success": True, "state": session.snapshot()}

    @tool
    async def live_view_get_state(self, view_id: str) -> dict:
        """Read the current state of an open LiveView.

        Returns the latest config and coordination space — including changes
        the *user* made by interacting with the component directly. Call this
        before deciding how to drive the view further.

        Args:
            view_id: id returned by open_live_view.

        Returns:
            dict with success and the current state snapshot.
        """
        try:
            session = self._require(view_id)
        except KeyError as e:
            return {"success": False, "error": str(e)}
        return {"success": True, "state": session.snapshot()}

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
        config: dict | None = None,
        coordination_space: dict | None = None,
        status: str | None = None,
        error: str | None = None,
    ) -> dict:
        """UI → backend: report the component's state after a user interaction.

        Called by the frontend whenever the user manipulates the component
        (pans, selects cells, …) or its lifecycle changes (ready / error), so
        the authoritative state — and thus the agent's next get_state — stays
        in sync with what the user sees.
        """
        session = self._views.get(view_id)
        if session is None:
            return {"success": False, "error": f"No LiveView with id '{view_id}'"}
        if config is not None:
            session.config = config
        if coordination_space is not None:
            session.coordination_space = coordination_space
        elif config is not None:
            session.coordination_space = config.get("coordinationSpace", {}) or {}
        if status is not None:
            session.status = status
        if error is not None:
            session.error = error
        session.updated_at = time.time()
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
