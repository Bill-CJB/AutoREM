"""OpenAI-compatible chat client configured exclusively through the environment."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


class LLMNetworkError(RuntimeError):
    """A transient transport or API availability error."""


class LLMAPIError(RuntimeError):
    """A non-retryable API configuration or response error."""


@dataclass
class LLMResponse:
    text: str
    latency: float = 0.0
    finish_reason: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)

    @property
    def prompt_tokens(self) -> int:
        return int(self.usage.get("prompt_tokens", 0) or 0)

    @property
    def completion_tokens(self) -> int:
        return int(self.usage.get("completion_tokens", 0) or 0)

    @property
    def total_tokens(self) -> int:
        return int(self.usage.get("total_tokens", self.prompt_tokens + self.completion_tokens) or 0)


class LLMClient:
    """A stateless client, safe to share across evaluation worker threads.

    Each role's request settings are selected by ``from_env``. API credentials
    are never written to logs or checkpoints.
    """

    def __init__(self, *, model: str, api_key: str, base_url: str,
                 timeout: float = 600.0, enable_thinking: bool | None = None,
                 reasoning_effort: str | None = None, max_tokens: int | None = 32768,
                 json_mode: bool = True):
        if not api_key or api_key == "YOUR_API_KEY":
            raise ValueError("Set AUTOREM_API_KEY to your API credential.")
        parts = urlsplit(base_url)
        if parts.username or parts.password or parts.query or parts.fragment:
            raise ValueError("API base URL must not contain credentials, a query, or a fragment.")
        if parts.scheme != "https" and not (
            parts.scheme == "http" and parts.hostname in ("localhost", "127.0.0.1", "::1")
        ):
            raise ValueError("Use HTTPS for remote API endpoints.")
        if not parts.hostname or not model or timeout <= 0 or (max_tokens is not None and max_tokens < 1):
            raise ValueError("Invalid API client configuration.")
        self.model = model
        self._api_key = api_key
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.timeout = timeout
        self.enable_thinking = enable_thinking
        self.reasoning_effort = reasoning_effort
        self.max_tokens = max_tokens
        self.json_mode = json_mode

    @classmethod
    def from_env(cls, *, coder: bool = False, task: str = "reformulation") -> "LLMClient":
        if task not in ("reformulation", "application", "training"):
            raise ValueError("Unknown API task.")
        prefix = "AUTOREM_CODER_" if coder else "AUTOREM_"
        def setting(name: str, default: str = "") -> str:
            return os.environ.get(prefix + name, default)
        uses_dashscope = not coder and task == "reformulation"
        default_url = ("https://dashscope.aliyuncs.com/compatible-mode/v1" if uses_dashscope
                       else "https://api.deepseek.com/v1")
        base_url = setting("BASE_URL", default_url)
        api_key = setting("API_KEY")
        # Reuse a credential only when both roles explicitly target the same URL.
        if coder and not api_key and base_url.rstrip("/") == os.environ.get("AUTOREM_BASE_URL", "").rstrip("/"):
            api_key = os.environ.get("AUTOREM_API_KEY", "")
        if not api_key:
            raise ValueError(f"Set {prefix}API_KEY for the selected API endpoint.")
        model = os.environ.get(prefix + "MODEL", "deepseek-v4-pro" if coder else "deepseek-v4-flash")
        thinking = setting("ENABLE_THINKING", "true" if uses_dashscope else "").lower()
        if thinking not in ("true", "false", ""):
            raise ValueError("ENABLE_THINKING must be true, false, or empty to omit the field.")
        max_tokens = setting("MAX_TOKENS", "" if coder else "32768")
        json_mode = setting("JSON_MODE", "false" if coder else "true").lower()
        if json_mode not in ("true", "false"):
            raise ValueError("JSON_MODE must be true or false.")
        return cls(
            model=model,
            api_key=api_key,
            base_url=base_url,
            timeout=float(setting("TIMEOUT", "600")),
            enable_thinking=None if thinking == "" else thinking == "true",
            reasoning_effort=setting("REASONING_EFFORT", "high" if uses_dashscope else "") or None,
            max_tokens=int(max_tokens) if max_tokens else None,
            json_mode=json_mode == "true",
        )

    def complete(self, prompt: str, **kwargs: Any) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a JSON-only assistant that MUST reply with a single valid JSON object without extra text."},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "temperature": 0,
        }
        for name in ("max_tokens", "enable_thinking", "reasoning_effort"):
            value = getattr(self, name)
            if value is not None:
                payload[name] = value
        if self.json_mode:
            payload["response_format"] = {"type": "json_object"}
        allowed = {"temperature", "max_tokens", "reasoning_effort", "enable_thinking"}
        if set(kwargs) - allowed:
            raise ValueError("Unsupported completion options.")
        payload.update(kwargs)
        request = Request(self.url, data=json.dumps(payload).encode("utf-8"), headers={
            "Authorization": "Bearer " + self._api_key,
            "Content-Type": "application/json",
        }, method="POST")
        start = time.perf_counter()
        try:
            with urlopen(request, timeout=self.timeout) as response:
                data = json.load(response)
        except HTTPError as exc:
            status = exc.code
            exc.close()
            # Response bodies may echo credentials; do not expose them.
            if status in (408, 429) or status >= 500:
                raise LLMNetworkError(f"API temporarily unavailable (HTTP {status}).") from None
            raise LLMAPIError(f"API request rejected (HTTP {status}); check endpoint, model, and credentials.") from None
        except (URLError, TimeoutError, OSError):
            raise LLMNetworkError("API connection failed or timed out.") from None
        except (ValueError, UnicodeError):
            raise LLMAPIError("API returned an invalid JSON response.") from None
        try:
            choice = data["choices"][0]
            content = choice["message"]["content"]
            if not isinstance(content, str):
                raise TypeError
            return LLMResponse(text=content, latency=time.perf_counter() - start,
                               finish_reason=choice.get("finish_reason"),
                               usage=data.get("usage") or {})
        except (KeyError, IndexError, TypeError):
            raise LLMAPIError("API response does not contain a text completion.") from None
