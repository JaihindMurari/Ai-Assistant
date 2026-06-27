"""
rag/llm.py
----------
Provider-agnostic LLM interface. Supports OpenAI and Google Gemini, selected
via the LLM_PROVIDER environment variable. Also provides token/cost
estimation utilities used by the "Estimated API Cost" bonus feature.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# Rough per-1K-token pricing (USD) for cost estimation only. Update as needed;
# this is intentionally approximate and clearly labeled as an estimate in the UI.
_PRICING_PER_1K = {
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gemini-1.5-flash": {"input": 0.000075, "output": 0.0003},
    "gemini-1.5-pro": {"input": 0.00125, "output": 0.005},
}


@dataclass
class LLMResponse:
    """Normalized response object regardless of provider."""

    text: str
    input_tokens: int
    output_tokens: int
    model: str
    provider: str

    @property
    def estimated_cost_usd(self) -> float:
        pricing = _PRICING_PER_1K.get(self.model, {"input": 0.0005, "output": 0.0015})
        cost = (self.input_tokens / 1000) * pricing["input"] + (self.output_tokens / 1000) * pricing["output"]
        return round(cost, 6)


class LLMError(Exception):
    """Raised when the LLM call fails (missing key, network error, API error)."""


class BaseLLMProvider:
    """Common interface every provider implementation must satisfy."""

    def generate(self, prompt: str, temperature: float, max_tokens: int) -> LLMResponse:
        raise NotImplementedError

    def count_tokens(self, text: str) -> int:
        raise NotImplementedError


class OpenAIProvider(BaseLLMProvider):
    """OpenAI GPT provider via the official SDK."""

    def __init__(self, api_key: str, model: str):
        if not api_key:
            raise LLMError("OpenAI API key is missing. Set OPENAI_API_KEY in your .env file.")
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate(self, prompt: str, temperature: float, max_tokens: int) -> LLMResponse:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            raise LLMError(f"OpenAI request failed: {exc}") from exc

        choice = response.choices[0].message.content or ""
        usage = response.usage
        return LLMResponse(
            text=choice.strip(),
            input_tokens=usage.prompt_tokens if usage else self.count_tokens(prompt),
            output_tokens=usage.completion_tokens if usage else self.count_tokens(choice),
            model=self.model,
            provider="openai",
        )

    def count_tokens(self, text: str) -> int:
        try:
            import tiktoken

            enc = tiktoken.encoding_for_model(self.model)
            return len(enc.encode(text))
        except Exception:
            return max(1, len(text) // 4)  # rough fallback estimate


class GeminiProvider(BaseLLMProvider):
    """Google Gemini provider via the google-generativeai SDK."""

    def __init__(self, api_key: str, model: str):
        if not api_key:
            raise LLMError("Google API key is missing. Set GOOGLE_API_KEY in your .env file.")
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        self.model_name = model
        self._client = genai.GenerativeModel(model)

    def generate(self, prompt: str, temperature: float, max_tokens: int) -> LLMResponse:
        try:
            response = self._client.generate_content(
                prompt,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                },
            )
        except Exception as exc:
            raise LLMError(f"Gemini request failed: {exc}") from exc

        text = (response.text or "").strip() if hasattr(response, "text") else ""
        usage = getattr(response, "usage_metadata", None)
        input_tokens = getattr(usage, "prompt_token_count", None) or self.count_tokens(prompt)
        output_tokens = getattr(usage, "candidates_token_count", None) or self.count_tokens(text)

        return LLMResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=self.model_name,
            provider="gemini",
        )

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)  # rough fallback estimate; Gemini SDK token counting needs a live call


def get_llm_provider(provider: str, openai_api_key: str, openai_model: str, google_api_key: str, gemini_model: str) -> BaseLLMProvider:
    """Factory function that returns the configured LLM provider instance."""
    provider = (provider or "").lower()
    if provider == "openai":
        return OpenAIProvider(api_key=openai_api_key, model=openai_model)
    if provider == "gemini":
        return GeminiProvider(api_key=google_api_key, model=gemini_model)
    raise LLMError(f"Unknown LLM provider: '{provider}'. Use 'openai' or 'gemini'.")
