"""Clase base abstracta que define qué debe saber hacer todo cliente de LLM.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from schemas import ChatMessage, ModelConfig, ModelResponse

# Constantes compartidas por los dos clientes para la lógica de reintentos.
MAX_REINTENTOS = 3  # reintentos adicionales tras el primer intento fallido
ESPERA_BASE_SEGUNDOS = 1  # backoff exponencial


class BaseLLMClient(ABC):
    """Interfaz común para todos los proveedores."""

    # Cada subclase define su nombre (se usa en ModelResponse.provider).
    provider_name: str = "base"

    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    @abstractmethod
    async def generate(self, messages: list[ChatMessage]) -> ModelResponse:
        """Envía los mensajes y devuelve la respuesta completa de una sola vez."""

    @abstractmethod
    def stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        """Devuelve la respuesta poco a poco (token a token).
        """
