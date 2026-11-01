"""Record safe run and API-response metadata for reproducibility."""

from __future__ import annotations

import json
import math
import re
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


API_CALL_FIELDS = (
    "request_started_utc", "response_received_utc", "requested_model",
    "returned_model", "api_response_created", "system_fingerprint",
)


def utc_now() -> str:
    """Return an explicit UTC timestamp independent of the host timezone."""
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def safe_identifier(value: Any, *, secret: str = "") -> str | None:
    """Accept a short model/fingerprint identifier without URLs or secrets."""
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}", value):
        return None
    if "://" in value or (secret and secret in value):
        return None
    return value


def api_call_metadata(*, request_started_utc: str, response_received_utc: str,
                      requested_model: str, response: dict, secret: str = "") -> dict:
    """Extract an explicit allowlist; never persist the response object."""
    created = response.get("created")
    if (isinstance(created, bool) or not isinstance(created, (int, float))
            or (isinstance(created, float) and not math.isfinite(created))
            or created < 0 or created > 2**63 - 1):
        created = None
    return {
        "request_started_utc": request_started_utc,
        "response_received_utc": response_received_utc,
        "requested_model": safe_identifier(requested_model, secret=secret),
        "returned_model": safe_identifier(response.get("model"), secret=secret),
        "api_response_created": created,
        "system_fingerprint": safe_identifier(response.get("system_fingerprint"), secret=secret),
    }


class _APICallLog:
    def __init__(self, path: Path):
        self._handle = path.open("w", encoding="utf-8", newline="\n")
        self._lock = threading.Lock()

    def write(self, record: dict) -> None:
        payload = {field: record.get(field) for field in API_CALL_FIELDS}
        line = json.dumps(payload, ensure_ascii=True, allow_nan=False) + "\n"
        with self._lock:
            self._handle.write(line)
            self._handle.flush()

    def close(self) -> None:
        with self._lock:
            self._handle.close()


class RunMetadata:
    """Record one execution and its successful API responses.

    The metadata files describe the current run in the supplied output directory.
    Reusing the directory replaces these files with records for that execution.
    """

    def __init__(self, directory: str | Path, *, base_client, coder_client):
        self.directory = Path(directory)
        self.base_client = base_client
        self.coder_client = coder_client
        self._logs: list[_APICallLog] = []
        self._previous_loggers: list[Any] = []
        self._metadata: dict[str, Any] = {}

    def _write_metadata(self) -> None:
        destination = self.directory / "run_metadata.json"
        temporary = self.directory / ".run_metadata.json.tmp"
        temporary.write_text(json.dumps(self._metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(destination)

    def __enter__(self) -> "RunMetadata":
        self.directory.mkdir(parents=True, exist_ok=True)
        self._metadata = {
            "run_started_utc": utc_now(),
            "run_finished_utc": None,
            "status": "running",
            "requested_models": {
                "base": self.base_client.safe_model_identifier(),
                "coder": self.coder_client.safe_model_identifier(),
            },
            "request_settings": {
                "base": self.base_client.safe_request_settings(),
                "coder": self.coder_client.safe_request_settings(),
            },
        }
        self._write_metadata()
        try:
            for role, client in (("base", self.base_client), ("coder", self.coder_client)):
                log = _APICallLog(self.directory / f"{role}_api_calls.jsonl")
                self._logs.append(log)
                self._previous_loggers.append(client.set_metadata_logger(log.write))
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        for client, previous in zip((self.base_client, self.coder_client), self._previous_loggers):
            client.set_metadata_logger(previous)
        for log in self._logs:
            log.close()
        self._metadata["run_finished_utc"] = utc_now()
        self._metadata["status"] = "failed" if exc_type is not None else "completed"
        self._write_metadata()
        return False
