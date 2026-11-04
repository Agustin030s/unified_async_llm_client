"""Cliente para Anthropic (Claude) usando el SDK asíncrono oficial (AsyncAnthropic)."""

import asyncio
from collections.abc import AsyncIterator

from anthropic import (
    APIConnectionError, 
    APIStatusError,
    AsyncAnthropic,
    AuthenticationError,
    RateLimitError,
)

from base_client import ESPERA_BASE_SEGUNDOS, MAX_REINTENTOS, BaseLLMClient
from schemas import ChatMessage, ModelConfig, ModelResponse


class AnthropicClient(BaseLLMClient):
    provider_name = "anthropic"

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

        # No enviamos `temperature`: los modelos actuales de Anthropic ya no aceptan ese parámetro.
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

    async def generate(self, messages: list[ChatMessage]) -> ModelResponse:
        ultimo_error = ""

        # range(MAX_REINTENTOS + 1) = 1 intento inicial + 3 reintentos.
        for intento in range(MAX_REINTENTOS + 1):
            try:
                response = await self.client.messages.create(**self._build_params(messages))
                # La respuesta es una lista de "bloques" de contenido; nos
                # quedamos con el texto de los bloques de tipo "text".
                texto = "".join(
                    block.text for block in response.content if block.type == "text"
                )
                return ModelResponse(
                    success=True, content=texto, provider=self.provider_name
                )

            except AuthenticationError:
                return ModelResponse(
                    success=False,
                    provider=self.provider_name,
                    error="API key de Anthropic inválida. Revisa ANTHROPIC_API_KEY en tu .env.",
                )

            except (RateLimitError, APIConnectionError) as e:
                ultimo_error = f"{type(e).__name__}: {e}"
                if intento < MAX_REINTENTOS:
                    espera = ESPERA_BASE_SEGUNDOS * 2**intento
                    print(
                        f"[anthropic] {type(e).__name__}, reintento "
                        f"{intento + 1}/{MAX_REINTENTOS} en {espera}s..."
                    )
                    await asyncio.sleep(espera)

            except APIStatusError as e:
                return ModelResponse(
                    success=False,
                    provider=self.provider_name,
                    error=f"Error de la API de Anthropic ({e.status_code}): {e.message}",
                )

            except Exception as e:
                return ModelResponse(
                    success=False,
                    provider=self.provider_name,
                    error=f"Error inesperado: {type(e).__name__}: {e}",
                )

        return ModelResponse(
            success=False,
            provider=self.provider_name,
            error=f"Falló tras {MAX_REINTENTOS} reintentos. Último error: {ultimo_error}",
        )

    async def stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        try:
            # `messages.stream` es un gestor de contexto asíncrono: `async with`
            # garantiza que la conexión se cierre aunque ocurra un error.
            async with self.client.messages.stream(**self._build_params(messages)) as s:
                # text_stream ya nos da solo el texto, sin eventos intermedios.
                async for text in s.text_stream:
                    yield text

        except AuthenticationError:
            yield "\n[ERROR] API key de Anthropic inválida. Revisa ANTHROPIC_API_KEY en tu .env."
        except (RateLimitError, APIConnectionError) as e:
            yield f"\n[ERROR] {type(e).__name__}: {e}"
        except APIStatusError as e:
            yield f"\n[ERROR] Error de la API de Anthropic ({e.status_code}): {e.message}"
        except Exception as e:
            yield f"\n[ERROR] Error inesperado: {type(e).__name__}: {e}"
