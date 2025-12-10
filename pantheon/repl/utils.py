"""Shared utilities for REPL UI components."""
import sys
from rich.console import Console
from rich.markdown import Markdown
from rich.box import Box
from typing import Any
from datetime import datetime


# Custom box style mimicking Claude Code (Rounded outer, disconnected inner)
# We replace T-junctions (├, ┤, ┬, ┴) with straight lines (│, ─)
CLAUDE_BOX = Box(
    "╭──╮\n"
    "│  │\n"
    "│──│\n"
    "│  │\n"
    "│──│\n"
    "│──│\n"
    "│  │\n"
    "╰──╯\n"
)

def get_animation_frames() -> list:
    """Get animation frames, with ASCII fallback for Windows.
    
    Returns:
        List of spinner frame characters.
    """
    fancy_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    ascii_frames = ["-", "\\", "|", "/"]
    
    try:
        # Check if stdout can handle unicode
        test_char = fancy_frames[0]
        if sys.stdout.encoding:
            test_char.encode(sys.stdout.encoding)
        return fancy_frames
    except (UnicodeEncodeError, LookupError, AttributeError):
        return ascii_frames

def get_separator() -> str:
    """Get separator character, with ASCII fallback.
    
    Returns:
        Separator character.
    """
    try:
        sep = "•"
        if sys.stdout.encoding:
            sep.encode(sys.stdout.encoding)
        return sep
    except (UnicodeEncodeError, LookupError, AttributeError):
        return "|"

def format_tool_name(tool_name: str) -> str:
    """Format tool name for status display: 'toolset__function' → 'function'.
    
    Args:
        tool_name: Raw tool name (potentially namespaced).
        
    Returns:
        Simplified tool name for display.
    """
    if not tool_name:
        return ""
    if "__" in tool_name:
        _, function = tool_name.split("__", 1)
        return function
    return tool_name

def format_relative_time(iso_time: str | datetime | None) -> str:
    """Format ISO time string to relative/friendly format.
    
    Args:
        iso_time: ISO format string or datetime object
        
    Returns:
        Formatted string (e.g. "Today 12:00", "5m ago")
    """
    if not iso_time:
        return "-"
    try:
        if isinstance(iso_time, str):
            dt = datetime.fromisoformat(iso_time)
        else:
            dt = iso_time
            
        now = datetime.now()
        diff = now - dt

        # Sub-minute fidelity for very recent items
        if diff.days == 0 and diff.seconds < 60:
            return "just now"
            
        # Standard relative formatting
        if diff.days == 0:
            return f"Today {dt.strftime('%H:%M')}"
        elif diff.days == 1:
            return f"Yesterday {dt.strftime('%H:%M')}"
        elif diff.days < 7:
            return dt.strftime("%a %H:%M")
        else:
            return dt.strftime("%b %d %H:%M")
    except Exception:
        return "-"

# Wave effect brightness levels (grey scale gradient)
# Using Hex codes for compatibility between Rich and prompt_toolkit
WAVE_COLORS = [
    "#4d4d4d", "#6b6b6b", "#8a8a8a", "#a8a8a8", "#c7c7c7", "#e3e3e3", 
    "#ffffff", 
    "#e3e3e3", "#c7c7c7", "#a8a8a8", "#8a8a8a", "#6b6b6b", "#4d4d4d"
]

def get_wave_color(index: int, offset: int) -> str:
    """Get color for wave animation at specific character index.
    
    Args:
        index: Character index in the string.
        offset: Animation frame offset.
        
    Returns:
        Hex color string.
    """
    pos = (index + offset) % len(WAVE_COLORS)
    return WAVE_COLORS[pos]


class OutputAdapter:
    """Unified output adapter for prompt_toolkit + Rich integration.
    
    When inside patch_stdout(raw=True) context, all output must go through
    sys.stdout to be correctly captured and rendered above the prompt.
    This adapter automatically switches between default Console and
    stdout-bound Console based on context.
    """
    
    def __init__(self):
        # Default Console for non-patch_stdout context (e.g., banner)
        self._default_console = Console()
        # Whether we're inside patch_stdout context
        self._in_patch_context = False
    
    @property
    def console(self) -> Console:
        """Get the appropriate Console based on current context.
        
        Returns:
            Console bound to sys.stdout when in patch_stdout context,
            otherwise returns the default Console.
        """
        if self._in_patch_context:
            # In patch_stdout, use sys.stdout Console for ANSI preservation
            return Console(file=sys.stdout, force_terminal=True)
        return self._default_console
    
    def enter_patch_context(self):
        """Called when entering patch_stdout context."""
        self._in_patch_context = True
    
    def exit_patch_context(self):
        """Called when exiting patch_stdout context."""
        self._in_patch_context = False
    
    def print(self, *args, **kwargs):
        """Print with automatic context-aware console selection."""
        self.console.print(*args, **kwargs)
    
    def print_markdown(self, content: str):
        """Print markdown content."""
        self.console.print(Markdown(content))


def format_token_count(count: int) -> str:
    """Format token count with K/M units."""
    if count >= 1_000_000: return f"{count/1_000_000:.1f}M"
    if count >= 10_000: return f"{count/1_000:.1f}K"
    return f"{count:,}" if count >= 1000 else str(count)


def get_token_stats(chatroom, chat_id, team, fallback: dict) -> dict:
    """Gather token statistics from chatroom or fallback to local stats."""
    from ..utils.llm import count_tokens_in_messages, process_messages_for_model
    from ..utils.log import logger
    
    messages, model = [], "unknown"
    
    # Get model from team first (needed for processing)
    if team and team.agents:
        agent = list(team.agents.values())[0]
        model = (agent.models[0] if isinstance(getattr(agent, 'models', None), list) 
                 else getattr(agent, 'models', None) or getattr(agent, 'model', 'unknown'))
    
    # Try to get messages from chatroom via memory_manager
    if chatroom and chat_id:
        try:
            if hasattr(chatroom, 'memory_manager'):
                memory = chatroom.memory_manager.get_memory(chat_id)
                if memory:
                    raw_messages = memory.get_messages(None) or []
                    messages = process_messages_for_model(raw_messages, model)
        except Exception as e:
            logger.warning(f"Failed to get messages for token stats: {e}")
    
    if messages:
        try:
            info = count_tokens_in_messages(messages, model)
            info["model"] = model
            return info
        except Exception as e:
            logger.warning(f"Failed to count tokens: {e}")
    
    # Fallback
    total = fallback.get("total_input_tokens", 0) + fallback.get("total_output_tokens", 0)
    return {
        "total": total, "max_tokens": 200000, "remaining": 200000 - total,
        "usage_percent": round(total / 200000 * 100, 1) if total else 0,
        "by_role": {"user": fallback.get("total_input_tokens", 0), "assistant": fallback.get("total_output_tokens", 0)},
        "message_counts": {"user": fallback.get("message_count", 0), "assistant": fallback.get("message_count", 0)},
        "warning_90": False, "critical_95": False, "current_cost": 0, "model": model,
    }


def render_token_panel(console: Console, info: dict, session_start: datetime):
    """Render Claude Code-style token analysis panel."""
    total = info.get("total", 0)
    by_role = info.get("by_role", {})
    msg_counts = info.get("message_counts", {})
    max_tok = info.get("max_tokens", 200000)
    usage_pct = info.get("usage_percent", 0)
    
    B = "[bold blue]"  # Box border color
    role_colors = {"system": "blue", "user": "green", "assistant": "yellow", "tool": "magenta"}
    
    console.print()
    console.print(f"{B}╭─ Context ─────────────────────────────────────────────────────────╮[/]")
    
    if total == 0:
        console.print(f"{B}│[/] [dim]No token usage data yet[/]")
        console.print(f"{B}╰───────────────────────────────────────────────────────────────────╯[/]")
        return
    
    # Build multi-color progress bar by role
    bar_w = 50
    used_ratio = total / max_tok if max_tok > 0 else 0
    used_width = max(1, round(used_ratio * bar_w)) if total > 0 else 0  # At least 1 block if any usage
    
    # Build colored segments proportional to each role
    bar_segments = []
    for role in ["system", "user", "assistant", "tool"]:
        role_tokens = by_role.get(role, 0)
        if role_tokens > 0 and total > 0:
            seg_width = max(1, round((role_tokens / total) * used_width))
            color = role_colors.get(role, "white")
            bar_segments.append(f"[{color}]{'█' * seg_width}[/]")
    
    # Combine segments and add remaining empty space
    bar = "".join(bar_segments)
    # Calculate actual filled width from segments (may exceed due to rounding)
    actual_filled = sum(max(1, round((by_role.get(r, 0) / total) * used_width)) 
                        for r in ["system", "user", "assistant", "tool"] 
                        if by_role.get(r, 0) > 0) if total > 0 else 0
    remaining_width = max(0, bar_w - actual_filled)
    bar += f"[dim]{'░' * remaining_width}[/]"
    
    max_disp = f"{max_tok // 1000}K"
    console.print(f"{B}│[/] {bar} {usage_pct}% of {max_disp}")
    console.print(f"{B}│[/] [dim]Used:[/] {format_token_count(total)} [dim]• Remaining:[/] {format_token_count(info.get('remaining', 0))}")
    
    # Token distribution legend
    console.print(f"{B}├───────────────────────────────────────────────────────────────────┤[/]")
    for role in ["system", "user", "assistant", "tool"]:
        if (tok := by_role.get(role, 0)) > 0:
            pct = tok / total * 100
            color = role_colors[role]
            console.print(f"{B}│[/] [{color}]●[/] {role.capitalize():<10} {format_token_count(tok):>8} ({pct:4.1f}%) [dim]{msg_counts.get(role, 0)} msgs[/]")
    
    # Session stats
    console.print(f"{B}├───────────────────────────────────────────────────────────────────┤[/]")
    dur = int((datetime.now() - session_start).total_seconds() / 60)
    model = info.get("model", "unknown")[:30]
    console.print(f"{B}│[/] [dim]Messages:[/] {sum(msg_counts.values())} [dim]• Duration:[/] {dur}m [dim]• Model:[/] {model}")
    if (cost := info.get("current_cost", 0)) > 0:
        console.print(f"{B}│[/] [dim]Estimated Cost:[/] ${cost:.4f}")
    
    # Warning
    if info.get("warning_90") or info.get("critical_95"):
        console.print(f"{B}├───────────────────────────────────────────────────────────────────┤[/]")
        if info.get("critical_95"):
            console.print(f"{B}│[/] [bold red]⚠ CRITICAL:[/] Context nearly full!")
        else:
            console.print(f"{B}│[/] [bold yellow]⚠ WARNING:[/] Context usage over 90%")
    
    console.print(f"{B}╰───────────────────────────────────────────────────────────────────╯[/]")
