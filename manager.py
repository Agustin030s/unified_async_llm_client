"""Punto único para obtener el cliente adecuado según la configuración del .env."""

import os

from anthropic_client import AnthropicClient
from base_client import BaseLLMClient
from openai_client import OpenAIClient
from schemas import ModelConfig

# Modelos por defecto si no se definen en el .env.
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5-5"

PROVEEDORES_VALIDOS = ("openai", "anthropic")


class AsyncLLMManager:
    """Lee la configuración del entorno y crea el cliente que corresponda.
    """

    def __init__(self) -> None:
        self.provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        self.openai_model = os.getenv("OPENAI_MODEL") or DEFAULT_OPENAI_MODEL
        self.anthropic_model = os.getenv("ANTHROPIC_MODEL") or DEFAULT_ANTHROPIC_MODEL

    def get_client(self, provider: str | None = None) -> BaseLLMClient:
        """Devuelve el cliente del proveedor indicado (o el de LLM_PROVIDER).

        Lanza ValueError si el proveedor no existe o falta su API key: es un
        error de configuración y conviene detectarlo antes de llamar a la API.
        """
        provider = (provider or self.provider).strip().lower()

        if provider == "openai":
            if not self.openai_api_key:
                raise ValueError("Falta OPENAI_API_KEY en el archivo .env")
            return OpenAIClient(ModelConfig(model=self.openai_model), self.openai_api_key)

        if provider == "anthropic":
            if not self.anthropic_api_key:
                raise ValueError("Falta ANTHROPIC_API_KEY en el archivo .env")
            return AnthropicClient(
                ModelConfig(model=self.anthropic_model), self.anthropic_api_key
            )

        raise ValueError(
            f"Proveedor '{provider}' no soportado. Usa uno de: {', '.join(PROVEEDORES_VALIDOS)}"
        )
