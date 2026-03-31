"""Structured logging setup with structlog, optional LangSmith tracing, and metrics."""

import os
import time
import logging
import threading
from functools import wraps

import structlog


def setup_logging(log_level: str | None = None):
    level = log_level or os.environ.get("LOG_LEVEL", "INFO")
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(format="%(message)s", level=numeric_level)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)


def setup_langsmith():
    """Enable LangSmith tracing if API key is set."""
    api_key = os.environ.get("LANGSMITH_API_KEY")
    if api_key:
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        os.environ.setdefault("LANGSMITH_PROJECT", "food-tracker")
        return True
    return False


class MetricsCollector:
    """Thread-safe metrics collector for tracking agent load and performance."""

    def __init__(self):
        self._lock = threading.Lock()
        self._total_requests = 0
        self._active_requests = 0
        self._total_errors = 0
        self._request_durations: list[float] = []
        self._max_duration_history = 1000
        self._start_time = time.time()
        self._requests_by_type: dict[str, int] = {}

    def record_request_start(self):
        with self._lock:
            self._total_requests += 1
            self._active_requests += 1

    def record_request_end(self, duration: float, request_type: str = "unknown"):
        with self._lock:
            self._active_requests = max(0, self._active_requests - 1)
            self._request_durations.append(duration)
            if len(self._request_durations) > self._max_duration_history:
                self._request_durations = self._request_durations[-self._max_duration_history:]
            self._requests_by_type[request_type] = self._requests_by_type.get(request_type, 0) + 1

    def record_error(self):
        with self._lock:
            self._total_errors += 1

    def get_metrics(self) -> dict:
        with self._lock:
            durations = self._request_durations.copy()
            avg_duration = sum(durations) / len(durations) if durations else 0
            p95_duration = sorted(durations)[int(len(durations) * 0.95)] if durations else 0

            return {
                "uptime_seconds": round(time.time() - self._start_time, 1),
                "total_requests": self._total_requests,
                "active_requests": self._active_requests,
                "total_errors": self._total_errors,
                "error_rate": round(self._total_errors / max(self._total_requests, 1), 4),
                "avg_response_time_ms": round(avg_duration * 1000, 1),
                "p95_response_time_ms": round(p95_duration * 1000, 1),
                "requests_by_type": dict(self._requests_by_type),
                "status": "healthy",
            }


metrics = MetricsCollector()


def track_request(request_type: str = "agent"):
    """Decorator to track request metrics."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            metrics.record_request_start()
            start = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            except Exception:
                metrics.record_error()
                raise
            finally:
                duration = time.time() - start
                metrics.record_request_end(duration, request_type)
        return wrapper
    return decorator
