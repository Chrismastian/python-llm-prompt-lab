"""IO layer: talk to an OpenAI-compatible /chat/completions endpoint.

Stdlib only (urllib, no SDK). No prompt wording lives here -> this file owns
retries, timeouts and HTTP errors, prompts.py owns the wording. Same split as
fetch.py vs parse.py in the scraper repo.
"""
import json
import os
import time
import urllib.error
import urllib.request

BASE_URL = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:20128/v1")
API_KEY = os.environ.get("LLM_API_KEY", "")
MODEL = os.environ.get("LLM_MODEL", "hermes")

# A 429 (rate limited) or 5xx (provider down) may succeed on a later attempt.
# A 401 (bad key) or 404 (bad model id) never will -> fail fast, do not burn time.
RETRY_STATUS = {429, 500, 502, 503, 504}


class LLMError(RuntimeError):
    """The endpoint rejected the call. Carries the status code for the caller."""

    def __init__(self, status: int, body: str):
        super().__init__(f"HTTP {status}: {body[:300]}")
        self.status = status
        self.body = body


def chat(
    messages: list[dict],
    model: str = MODEL,
    temperature: float = 0.0,
    attempts: int = 3,
    timeout: int = 180,
) -> dict:
    """POST the whole history and return the raw JSON reply.

    temperature=0.0 by default: prompt quality should be measured on a model
    that answers the same way twice, otherwise you cannot tell whether your
    prompt changed or the dice did.
    """
    payload = json.dumps(
        {"model": model, "messages": messages, "temperature": temperature}
    ).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        },
        method="POST",
    )
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as err:
            body = err.read().decode(errors="replace")
            if err.code not in RETRY_STATUS or attempt == attempts:
                raise LLMError(err.code, body) from err
            time.sleep(2**attempt)
        except (urllib.error.URLError, TimeoutError) as err:
            if attempt == attempts:
                raise LLMError(0, str(err)) from err
            time.sleep(2**attempt)
    raise LLMError(0, "unreachable")  # loop covers every path; keeps type checkers quiet