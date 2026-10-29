"""Production logging setup — structured JSON to stderr.

One JSON object per line on stderr, so `docker logs` and any log collector
(Loki, Datadog, ELK) can ingest and filter by level, service, tenant_id or
error_code without parsing free text. Request-level tracing lives in
Langfuse; these logs are for service health and errors.
"""
from __future__ import annotations

import logging
import sys

from backend.logging.base import LogSetup
from backend.logging.formatters import JSONFormatter

_NOISY_LOGGERS = ("httpx", "httpcore", "openai", "asyncio", "urllib3")


class ProdLogSetup(LogSetup):
    """JSON-to-stderr handler for containerised deployments."""

    def configure(self, service_name: str) -> None:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(JSONFormatter(service_name))
        handler.setLevel(logging.INFO)

        root = logging.getLogger()
        root.handlers.clear()
        root.addHandler(handler)
        root.setLevel(logging.INFO)
        logging.getLogger("backend").setLevel(logging.INFO)

        for name in _NOISY_LOGGERS:
            logging.getLogger(name).setLevel(logging.WARNING)
