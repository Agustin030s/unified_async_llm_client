"""Punto único para obtener el cliente adecuado según la configuración del .env."""

import os

from anthropic_client import AnthropicClient
from base_client import BaseLLMClient
from openai_client import OpenAIClient
from schemas import ModelConfig

# Registro de proveedores: {"openai": OpenAIClient, "anthropic": AnthropicClient}.
# Para agregar un proveedor nuevo basta con crear su clase y sumarla a esta
# tupla; el manager deriva de `provider_name` las variables de entorno que lee.
PROVEEDORES: dict[str, type[BaseLLMClient]] = {
    cls.provider_name: cls for cls in (OpenAIClient, AnthropicClient)
}


class AsyncLLMManager:
    """Lee la configuración del entorno y crea el cliente que corresponda.

    Nota: llama a load_dotenv() ANTES de usar el manager (lo hacemos en main.py).
    """

    def __init__(self) -> None:
        self.provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()

    def _build_config(self, client_class: type[BaseLLMClient]) -> ModelConfig:
        # Variables por proveedor: <PROVEEDOR>_MODEL (p. ej. OPENAI_MODEL).
        prefix = client_class.provider_name.upper()
        data = {"model": os.getenv(f"{prefix}_MODEL") or client_class.default_model}

        # Variables comunes a todos los proveedores. Si no están, ModelConfig
        # usa sus valores por defecto (0.7 y 512). Pydantic convierte el texto
        # del .env a float/int y valida los rangos.
        if temperature := os.getenv("LLM_TEMPERATURE"):
            data["temperature"] = temperature
        if max_tokens := os.getenv("LLM_MAX_TOKENS"):
            data["max_tokens"] = max_tokens
        return ModelConfig(**data)

    def get_client(self, provider: str | None = None) -> BaseLLMClient:
        """Devuelve el cliente del proveedor indicado (o el de LLM_PROVIDER).

        Lanza ValueError si el proveedor no existe, falta su API key o la
        configuración es inválida: son errores de configuración y conviene
        detectarlos antes de llamar a la API.
        """
        provider = (provider or self.provider).strip().lower()

        client_class = PROVEEDORES.get(provider)
        if client_class is None:
            raise ValueError(
                f"Proveedor '{provider}' no soportado. Usa uno de: {', '.join(PROVEEDORES)}"
            )

        api_key_var = f"{provider.upper()}_API_KEY"
        api_key = os.getenv(api_key_var, "").strip()
        if not api_key:
            raise ValueError(f"Falta {api_key_var} en el archivo .env")

        # Si LLM_TEMPERATURE o LLM_MAX_TOKENS son inválidos, Pydantic lanza
        # ValidationError, que es subclase de ValueError.
        return client_class(self._build_config(client_class), api_key)
