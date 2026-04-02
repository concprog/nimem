import logging
from typing import Any, Callable, TypeVar, Optional

from returns.result import Result, Success, Failure

logger = logging.getLogger(__name__)

T = TypeVar("T")
E = TypeVar("E")


def unwrap_result(
    result: Result[T, E],
    error_msg: str = "Operation failed",
    log_level: int = logging.WARNING,
) -> Optional[T]:
    """Safely unwrap a returns Result, logging and returning None on failure."""
    if isinstance(result, Success):
        return result.unwrap()
    elif isinstance(result, Failure):
        logger.log(log_level, f"{error_msg}: {result.failure()}")
    else:
        logger.log(log_level, f"{error_msg}: unknown result type {type(result)}")
    return None


def unwrap_or_else(
    result: Result[T, E],
    fallback: Callable[[], T],
    error_msg: str = "Operation failed",
    log_level: int = logging.DEBUG,
) -> T:
    """Unwrap Result, or call fallback on failure."""
    if isinstance(result, Success):
        return result.unwrap()
    else:
        logger.log(
            log_level,
            f"{error_msg}: {result.failure() if isinstance(result, Failure) else 'unknown'}",
        )
        return fallback()


def result_to_optional(result: Result[T, E]) -> Optional[T]:
    """Convert Result to Optional (None on Failure)."""
    if isinstance(result, Success):
        return result.unwrap()
    return None


def result_to_tuple(
    result: Result[T, E],
) -> tuple[bool, Any]:
    """Convert Result to (success, value_or_error)."""
    if isinstance(result, Success):
        return True, result.unwrap()
    return False, result.failure()
