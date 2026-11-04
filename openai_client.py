"""Cliente para OpenAI usando el SDK asíncrono oficial (AsyncOpenAI)."""

import asyncio
from collections.abc import AsyncIterator

from openai import (
    APIConnectionError,  # también cubre APITimeoutError (es una subclase)
    APIStatusError,
    AsyncOpenAI,
    AuthenticationError,
    RateLimitError,
)

from base_client import ESPERA_BASE_SEGUNDOS, MAX_REINTENTOS, BaseLLMClient
from schemas import ChatMessage, ModelConfig, ModelResponse


class OpenAIClient(BaseLLMClient):
    provider_name = "openai"

    def __init__(self, config: ModelConfig, api_key: str) -> None:
        super().__init__(config)
        # max_retries=0 desactiva los reintentos internos del SDK para que la
        # única lógica de reintentos sea la nuestra
        self.client = AsyncOpenAI(api_key=api_key, max_retries=0)

    def _to_openai_messages(self, messages: list[ChatMessage]) -> list[dict]:
        # OpenAI acepta directamente el formato {"role": ..., "content": ...},
        # incluido el rol "system", así que basta con convertir a diccionario.
        return [m.model_dump() for m in messages]

    async def generate(self, messages: list[ChatMessage]) -> ModelResponse:
        ultimo_error = ""

        # range(MAX_REINTENTOS + 1) = 1 intento inicial + 3 reintentos.
        for intento in range(MAX_REINTENTOS + 1):
            try:
                response = await self.client.chat.completions.create(
                    model=self.config.model,
                    messages=self._to_openai_messages(messages),
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                )
                return ModelResponse(
                    success=True,
                    content=response.choices[0].message.content or "",
                    provider=self.provider_name,
                )

            except AuthenticationError:
                # Una API key inválida no se arregla reintentando: salimos ya.
                return ModelResponse(
                    success=False,
                    provider=self.provider_name,
                    error="API key de OpenAI inválida. Revisa OPENAI_API_KEY en tu .env.",
                )

            except (RateLimitError, APIConnectionError) as e:
                # Errores temporales: tiene sentido esperar y volver a intentar.
                ultimo_error = f"{type(e).__name__}: {e}"
                if intento < MAX_REINTENTOS:
                    espera = ESPERA_BASE_SEGUNDOS * 2**intento
                    print(
                        f"[openai] {type(e).__name__}, reintento "
                        f"{intento + 1}/{MAX_REINTENTOS} en {espera}s..."
                    )
                    # asyncio.sleep NO bloquea el event loop (time.sleep sí lo haría).
                    await asyncio.sleep(espera)

            except APIStatusError as e:
                # Otros errores HTTP (modelo inexistente, petición mal formada...).
                return ModelResponse(
                    success=False,
                    provider=self.provider_name,
                    error=f"Error de la API de OpenAI ({e.status_code}): {e.message}",
                )

            except Exception as e:
                # Capturamos cualquier otra excepción para que no rompa el flujo de la app
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
            response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=self._to_openai_messages(messages),
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                stream=True,
            )
            async for chunk in response:
                # Algunos chunks no traen texto (p. ej. el último, o vienen sin
                # `choices`), por eso comprobamos antes de hacer yield.
                if chunk.choices and chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content

        # En streaming no podemos devolver un ModelResponse, así que
        # informamos del error como un último "trozo" de texto.
        except AuthenticationError:
            yield "\n[ERROR] API key de OpenAI inválida. Revisa OPENAI_API_KEY en tu .env."
        except (RateLimitError, APIConnectionError) as e:
            yield f"\n[ERROR] {type(e).__name__}: {e}"
        except APIStatusError as e:
            yield f"\n[ERROR] Error de la API de OpenAI ({e.status_code}): {e.message}"
        except Exception as e:
            yield f"\n[ERROR] Error inesperado: {type(e).__name__}: {e}"
