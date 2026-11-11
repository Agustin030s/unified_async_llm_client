"""Cliente para OpenAI usando el SDK asíncrono oficial (AsyncOpenAI).

Solo contiene lo específico de OpenAI; los reintentos y el manejo de errores
están en BaseLLMClient.
"""

from collections.abc import AsyncIterator

from openai import (
    APIConnectionError,  # también cubre APITimeoutError (es una subclase)
    AsyncOpenAI,
    AuthenticationError,
    RateLimitError,
)

from base_client import BaseLLMClient
from schemas import ChatMessage, ModelConfig


class OpenAIClient(BaseLLMClient):
    provider_name = "openai"
    # Verifica el nombre del modelo en https://platform.openai.com/docs/models
    default_model = "gpt-4o-mini"
    transient_errors = (RateLimitError, APIConnectionError)
    auth_errors = (AuthenticationError,)

    def __init__(self, config: ModelConfig, api_key: str) -> None:
        super().__init__(config)
        # max_retries=0 desactiva los reintentos internos del SDK para que la
        # única lógica de reintentos sea la nuestra
        self.client = AsyncOpenAI(api_key=api_key, max_retries=0)

    def _build_params(self, messages: list[ChatMessage]) -> dict:
        return {
            "model": self.config.model,
            # OpenAI acepta directamente el formato {"role": ..., "content": ...},
            # incluido el rol "system", así que basta con convertir a diccionario.
            "messages": [m.model_dump() for m in messages],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

    async def _request(self, messages: list[ChatMessage]) -> str:
        response = await self.client.chat.completions.create(**self._build_params(messages))
        return response.choices[0].message.content or ""

    async def _request_stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        response = await self.client.chat.completions.create(
            **self._build_params(messages), stream=True
        )
        async for chunk in response:
            # Algunos chunks no traen texto (p. ej. el último, o vienen sin
            # `choices`), por eso comprobamos antes de hacer yield.
            if chunk.choices and chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
