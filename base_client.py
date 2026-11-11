"""Clase base abstracta que define qué debe saber hacer todo cliente de LLM.

Aquí vive TODA la lógica común (reintentos, manejo de errores, streaming), de
modo que cada proveedor solo implementa lo que realmente cambia entre SDKs:
cómo hacer la petición y qué excepciones son transitorias.
"""

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable

from schemas import ChatMessage, ModelConfig, ModelResponse, StreamChunk

# Política de reintentos: si se cambia aquí, se aplica a todos los proveedores.
MAX_REINTENTOS = 3  # reintentos adicionales tras el primer intento fallido
ESPERA_BASE_SEGUNDOS = 1  # backoff exponencial: 1s, 2s, 4s


class BaseLLMClient(ABC):
    """Interfaz común para todos los proveedores."""

    # --- Lo que cada subclase debe declarar ---
    provider_name: str = "base"  # se usa en ModelResponse.provider y en el .env
    default_model: str = ""  # modelo si no se define <PROVEEDOR>_MODEL en el .env
    # Errores temporales: vale la pena esperar y reintentar (rate limit, red...).
    transient_errors: tuple[type[Exception], ...] = ()
    # Errores de autenticación: reintentar no los arregla.
    auth_errors: tuple[type[Exception], ...] = ()

    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    # --- Lo que cada subclase debe implementar (lo específico de cada SDK) ---

    @abstractmethod
    async def _request(self, messages: list[ChatMessage]) -> str:
        """Hace UNA petición al proveedor y devuelve el texto completo.

        No captura errores: los deja subir para que la clase base decida si
        reintentar o no.
        """

    @abstractmethod
    def _request_stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        """Abre UN stream con el proveedor y hace `yield` de cada trozo de texto.

        Tampoco captura errores. Las subclases lo implementan con `async def` y
        `yield` (un generador asíncrono).
        """

    # --- Lógica común a todos los proveedores ---

    def _describe_error(self, error: Exception) -> str:
        """Convierte una excepción en un mensaje legible."""
        if isinstance(error, self.auth_errors):
            return (
                f"API key de {self.provider_name} inválida. "
                f"Revisa {self.provider_name.upper()}_API_KEY en tu .env."
            )
        return f"{type(error).__name__}: {error}"

    async def _wait_before_retry(self, intento: int, error: Exception) -> None:
        espera = ESPERA_BASE_SEGUNDOS * 2**intento
        print(
            f"[{self.provider_name}] {type(error).__name__}, reintento "
            f"{intento + 1}/{MAX_REINTENTOS} en {espera}s..."
        )
        # asyncio.sleep cede el control al event loop; time.sleep lo bloquearía.
        await asyncio.sleep(espera)

    async def _with_retries(self, operation: Callable[[], Awaitable[str]]) -> str:
        """Ejecuta `operation` y la reintenta solo si falla con un error transitorio.

        Recibe una función (no su resultado) porque cada reintento necesita
        crear una petición NUEVA: una corrutina ya esperada no se puede reusar.
        """
        # range(MAX_REINTENTOS + 1) = 1 intento inicial + 3 reintentos.
        for intento in range(MAX_REINTENTOS + 1):
            try:
                return await operation()
            except self.transient_errors as e:
                if intento == MAX_REINTENTOS:
                    raise  # se agotaron los reintentos: que lo maneje quien llama
                await self._wait_before_retry(intento, e)
        raise AssertionError("inalcanzable")  # el bucle siempre retorna o lanza

    async def generate(self, messages: list[ChatMessage]) -> ModelResponse:
        """Envía los mensajes y devuelve la respuesta completa de una sola vez."""
        try:
            text = await self._with_retries(lambda: self._request(messages))
            return ModelResponse(success=True, content=text, provider=self.provider_name)
        except Exception as e:
            # Ningún error rompe el programa: lo devolvemos de forma estructurada.
            return ModelResponse(
                success=False, provider=self.provider_name, error=self._describe_error(e)
            )

    async def stream(self, messages: list[ChatMessage]) -> AsyncIterator[StreamChunk]:
        """Devuelve la respuesta poco a poco como objetos StreamChunk.

        Si falla con un error transitorio ANTES del primer trozo, reintenta. Si
        ya se emitió texto no reintentamos: el consumidor lo vería repetido.
        """
        for intento in range(MAX_REINTENTOS + 1):
            emitted = False
            try:
                async for text in self._request_stream(messages):
                    emitted = True
                    yield StreamChunk(text=text)
                return  # el stream terminó bien

            except self.transient_errors as e:
                if emitted or intento == MAX_REINTENTOS:
                    yield StreamChunk(error=self._describe_error(e))
                    return
                await self._wait_before_retry(intento, e)

            except Exception as e:
                # Errores no transitorios (auth, modelo inexistente...): sin reintento.
                yield StreamChunk(error=self._describe_error(e))
                return
