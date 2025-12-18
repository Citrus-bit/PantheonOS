import sys
import warnings
from contextlib import contextmanager
from loguru import logger as loguru_logger

LEVEL_MAP = {
    "DEBUG": 10,
    "INFO": 20,
    "WARNING": 30,
    "ERROR": 40,
}


@contextmanager
def temporary_log_level(level: str):
    """Context manager to temporarily set log level for loguru logger

    Usage:
        with temporary_log_level("WARNING"):
            agent.run()  # Only WARNING and ERROR will be logged
    """
    # Use loguru's contextualize to set a context variable
    # Then the filter checks this variable to decide whether to log
    with loguru_logger.contextualize(log_level_override=level):
        yield


# Configure loguru handler with context-aware filter
def _context_aware_filter(record):
    """Filter that respects context-local log level settings"""
    override_level = record["extra"].get("log_level_override")
    if override_level is None:
        return True  # No override, allow all logs

    override_level_num = LEVEL_MAP.get(override_level, 0)
    record_level_num = record["level"].no
    return record_level_num >= override_level_num


logger = loguru_logger

# Track if logging has been explicitly disabled
_logging_disabled = False

# Apply context-aware filter to all handlers
# Remove default handler and add new one with our filter
# Use stdout instead of stderr so it works with prompt_toolkit's patch_stdout
loguru_logger.remove()
loguru_logger.add(sys.stdout, filter=_context_aware_filter, level="WARNING")


def set_level(level: str):
    """Set the logging level."""
    global _logging_disabled
    if _logging_disabled:
        return  # Don't re-enable if disabled
    loguru_logger.remove()
    loguru_logger.add(sys.stdout, filter=_context_aware_filter, level=level)


def disable_all():
    """Completely disable all logging. Cannot be re-enabled."""
    global _logging_disabled
    _logging_disabled = True
    loguru_logger.remove()
    loguru_logger.disable("pantheon")


# =============================================================================
# Warning Suppression
# =============================================================================

# Suppress aiohttp "Unclosed client session" warnings from litellm.
# These warnings are harmless - the OS cleans up connections on process exit.
warnings.filterwarnings("ignore", message="Unclosed client session", category=ResourceWarning)
warnings.filterwarnings("ignore", message="Unclosed connector", category=ResourceWarning)


def suppress_aiohttp_warnings(loop, context) -> None:
    """Custom asyncio exception handler to suppress aiohttp cleanup warnings.
    
    aiohttp prints warnings via asyncio's exception handler during GC.
    Use with: loop.set_exception_handler(suppress_aiohttp_warnings)
    """
    message = context.get("message", "")
    if "Unclosed" in message:
        return  # Suppress aiohttp cleanup warnings
    # For other exceptions, use default handling
    loop.default_exception_handler(context)
