# python-llm-prompt-lab

[![CI](https://github.com/Chrismastian/python-llm-prompt-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/Chrismastian/python-llm-prompt-lab/actions/workflows/ci.yml)

Experiments in prompt engineering against an OpenAI-compatible LLM API, using
the standard library only (no SDK).

## What it does

- Builds the `messages` history the API expects and understands why the API is
  stateless (the whole history is resent on every call).
- Compares prompt variants (system vs user prompt, explicit output contract)
  and measures the result instead of guessing.
- Parses structured output out of replies that are wrapped in markdown fences.

## Layout

```
llmlab/prompts.py   pure logic: build messages, extract JSON  (no network)
llmlab/client.py    IO layer: HTTP call to /chat/completions  (no logic)
llmlab/cli.py       entry point
tests/              pytest suite for the pure layer
```

## How to run

```
python -m llmlab.cli "Explain what a token is in one sentence"
```

Configuration comes from environment variables, never from the code:

| Variable | Default | Meaning |
|---|---|---|
| `LLM_BASE_URL` | `http://127.0.0.1:20128/v1` | OpenAI-compatible endpoint |
| `LLM_API_KEY` | *(empty)* | Bearer token; required even on localhost |
| `LLM_MODEL` | `hermes` | Model id, taken from `GET /models` |

## Tests

```
python -m pytest
```