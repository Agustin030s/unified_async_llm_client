"""Cliente para Anthropic (Claude) usando el SDK asíncrono oficial (AsyncAnthropic).

Solo contiene lo específico de Anthropic; los reintentos y el manejo de errores
están en BaseLLMClient.
"""

from collections.abc import AsyncIterator

from anthropic import (
    APIConnectionError,  # también cubre APITimeoutError (es una subclase)
    AsyncAnthropic,
    AuthenticationError,
    RateLimitError,
)

from base_client import BaseLLMClient
from schemas import ChatMessage, ModelConfig


class AnthropicClient(BaseLLMClient):
    provider_name = "anthropic"
    # Verifica el nombre del modelo en https://docs.anthropic.com/en/docs/about-claude/models
    default_model = "claude-sonnet-5-5"
    transient_errors = (RateLimitError, APIConnectionError)
    auth_errors = (AuthenticationError,)

    def __init__(self, config: ModelConfig, api_key: str) -> None:
        super().__init__(config)
        # max_retries=0: desactivamos los reintentos internos del SDK para que
        # solo actúe nuestra lógica de reintentos.
        self.client = AsyncAnthropic(api_key=api_key, max_retries=0)

    def _build_params(self, messages: list[ChatMessage]) -> dict:
        """Traduce nuestro formato común al formato que espera Anthropic."""
        # Diferencia con OpenAI: Anthropic NO acepta el rol "system"
        # dentro de `messages`. El prompt de sistema va en un parámetro aparte.
        # Por eso separamos los mensajes en dos grupos.
        system_parts = [m.content for m in messages if m.role == "system"]
        chat_messages = [
            {"role": m.role, "content": m.content}
            for m in messages
            if m.role != "system"
        ]

        # No enviamos `temperature`: los modelos actuales de Anthropic ya no
        # aceptan ese parámetro, así que config.temperature se ignora aquí.
        params = {
            "model": self.config.model,
            "messages": chat_messages,
            # En Anthropic max_tokens es OBLIGATORIO (en OpenAI es opcional).
            "max_tokens": self.config.max_tokens,
        }
        # Solo enviamos `system` si existe; si hubiera varios, los unimos.
        if system_parts:
            params["system"] = "\n\n".join(system_parts)
        return params

    async def _request(self, messages: list[ChatMessage]) -> str:
        response = await self.client.messages.create(**self._build_params(messages))
        # La respuesta es una lista de "bloques" de contenido; nos quedamos con
        # el texto de los bloques de tipo "text".
        return "".join(block.text for block in response.content if block.type == "text")

    async def _request_stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        # `messages.stream` es un gestor de contexto asíncrono: `async with`
        # garantiza que la conexión se cierre aunque ocurra un error.
        async with self.client.messages.stream(**self._build_params(messages)) as s:
            # text_stream ya nos da solo el texto, sin eventos intermedios.
            async for text in s.text_stream:
                yield text
