"""Structured logging configuration."""

import logging
import sys

import structlog


def configure_logging(
    level: str = "INFO",
    json_format: bool = False,
) -> structlog.stdlib.BoundLogger:
    """Configure structured logging.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR).
        json_format: If True, output JSON logs. Otherwise, use console format.

    Returns:
        Configured logger instance.
    """
    # Set up standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper()),
    )

    # Configure structlog processors
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if json_format:
        # JSON output for production
        shared_processors.append(structlog.processors.JSONRenderer())
    else:
        # Console output for development
        shared_processors.append(
            structlog.dev.ConsoleRenderer(
                colors=True,
                exception_formatter=structlog.dev.plain_traceback,
            )
        )

    structlog.configure(
        processors=shared_processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    return structlog.get_logger()


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Get a logger instance.

    Args:
        name: Logger name. Defaults to root logger.

    Returns:
        Logger instance.
    """
    return structlog.get_logger(name)
