"""
tools/llm_client.py
─────────────────────────────────────────────────────────────────────────────
Shared LLM client with automatic key rotation and provider fallback.

Priority order:
  1. GROQ        (5 keys, fast, free tier)  → llama-3.3-70b-versatile
  2. GEMINI      (6 keys)                    → gemini-2.0-flash
  3. DEEPSEEK    (1 key)                     → deepseek-chat
  4. NVIDIA NIM  (1 key)                     → meta/llama-3.3-70b-instruct
  5. OpenAI      (1 key)                     → gpt-4o-mini
  6. Ollama      (local fallback)            → llama3:8b-instruct-q4_K_M

Key rotation:
  - Each provider's keys are read from comma-separated env vars
  - On RateLimitError: rotate to next key in same provider
  - On all keys exhausted: fall to next provider
  - On ConnectionError/APIError: skip provider immediately

Usage:
    from tools.llm_client import call_llm

    response = call_llm(messages, temperature=0.8)
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import os
import time
from typing import Any

import structlog
from openai import APIError, OpenAI, RateLimitError

log = structlog.get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Provider definitions
# ─────────────────────────────────────────────────────────────────────────────

def _load_keys(env_plural: str, env_singular: str) -> list[str]:
    """Read comma-separated key list from env, falling back to singular form."""
    raw = os.getenv(env_plural) or os.getenv(env_singular) or ""
    return [k.strip() for k in raw.split(",") if k.strip()]


def _build_providers() -> list[dict[str, Any]]:
    """Build ordered list of providers with their keys and config."""
    return [
        {
            "name": "groq",
            "keys": _load_keys("GROQ_API_KEYS", "GROQ_API_KEY"),
            "base_url": "https://api.groq.com/openai/v1",
            "model": "llama-3.3-70b-versatile",
        },
        {
            "name": "gemini",
            "keys": _load_keys("GEMINI_API_KEYS", "GEMINI_API_KEY"),
            "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
            "model": "gemini-2.0-flash",
        },
        {
            "name": "deepseek",
            "keys": _load_keys("DEEPSEEK_API_KEYS", "DEEPSEEK_API_KEY"),
            "base_url": "https://api.deepseek.com/v1",
            "model": "deepseek-chat",
        },
        {
            "name": "nvidia",
            "keys": _load_keys("NVIDIA_API_KEYS", "NVIDIA_API_KEY"),
            "base_url": "https://integrate.api.nvidia.com/v1",
            "model": "meta/llama-3.3-70b-instruct",
        },
        {
            "name": "openai",
            "keys": _load_keys("OPENAI_API_KEYS", "OPENAI_API_KEY"),
            "base_url": None,
            "model": "gpt-4o-mini",
        },
        {
            "name": "ollama",
            "keys": ["ollama"],           # always available as last resort
            "base_url": "http://localhost:11434/v1",
            "model": "llama3:8b-instruct-q4_K_M",
        },
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Core call with rotation
# ─────────────────────────────────────────────────────────────────────────────

def call_llm(
    messages: list[dict],
    temperature: float = 0.8,
    max_tokens: int = 4096,
    model_override: str | None = None,
) -> str:
    """
    Call LLM with automatic key rotation and provider fallback.

    Tries every key in every provider until one succeeds.
    Returns the response text.
    Raises RuntimeError if all providers fail.
    """
    providers = _build_providers()

    for provider in providers:
        keys     = provider["keys"]
        name     = provider["name"]
        base_url = provider["base_url"]
        model    = model_override or provider["model"]

        if not keys:
            continue

        for i, key in enumerate(keys):
            try:
                client = OpenAI(
                    base_url=base_url,
                    api_key=key,
                ) if base_url else OpenAI(api_key=key)

                log.info("llm.attempt", provider=name, key_index=i, model=model)

                resp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                content = resp.choices[0].message.content or ""
                log.info("llm.success", provider=name, key_index=i,
                          chars=len(content))
                return content

            except RateLimitError as e:
                log.warning("llm.rate_limit",
                            provider=name, key_index=i,
                            total_keys=len(keys),
                            msg=str(e)[:120])
                # Try next key in same provider
                time.sleep(1)
                continue

            except APIError as e:
                err_str = str(e)
                log.warning("llm.api_error",
                            provider=name, key_index=i, error=err_str[:120])
                # Connection errors → skip whole provider
                if "Connection" in err_str or "connect" in err_str.lower():
                    log.warning("llm.skipping_provider",
                                provider=name, reason="connection_error")
                    break
                # Other API errors → try next key
                time.sleep(2)
                continue

            except Exception as e:
                log.warning("llm.unexpected_error",
                            provider=name, key_index=i, error=str(e)[:120])
                break

        log.warning("llm.provider_exhausted", provider=name)

    raise RuntimeError(
        "All LLM providers exhausted. Check API keys in .env and internet access."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: get a (client, model) tuple for agents that need direct access
# ─────────────────────────────────────────────────────────────────────────────

def get_llm_client() -> tuple[OpenAI, str]:
    """
    Return the first available (OpenAI client, model_name).
    Used by agents that need the client object directly.
    Same priority order as call_llm().
    """
    providers = _build_providers()
    for provider in providers:
        keys = provider["keys"]
        if not keys:
            continue
        name     = provider["name"]
        base_url = provider["base_url"]
        model    = provider["model"]
        key      = keys[0]
        log.info("llm.client_selected", provider=name, model=model)
        client = OpenAI(
            base_url=base_url, api_key=key
        ) if base_url else OpenAI(api_key=key)
        return client, model

    log.warning("llm.all_providers_empty_using_ollama")
    return OpenAI(base_url="http://localhost:11434/v1", api_key="ollama"), \
           "llama3:8b-instruct-q4_K_M"
