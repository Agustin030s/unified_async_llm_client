"""
Pydantic nos da validación automática: si alguien crea un mensaje con un
rol inválido o una temperatura fuera de rango, el error aparece en el momento
de construir el objeto y no más tarde, dentro de una llamada a la API.
"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ChatMessage(BaseModel):
    """Un mensaje de la conversación, con el formato común a ambos proveedores."""

    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)

    @field_validator("content")
    @classmethod
    def no_solo_espacios(cls, value: str) -> str:
        # min_length=1 acepta "   ", pero un mensaje solo con espacios no tiene
        # sentido y algunas APIs lo rechazan, así que lo validamos aquí.
        if not value.strip():
            raise ValueError("El contenido del mensaje no puede estar vacío")
        return value


class ModelConfig(BaseModel):
    """Configuración del modelo que se usará en cada petición."""

    model: str
    # 0 = respuestas más deterministas, 2 = más creativas/aleatorias.
    temperature: float = Field(0.7, ge=0, le=2)
    max_tokens: int = Field(512, gt=0)


class ModelResponse(BaseModel):
    """Respuesta unificada: el resto del programa no necesita saber qué SDK se usó.

    En lugar de lanzar excepciones, los clientes devuelven success=False y un
    mensaje en `error`. Así quien llama solo tiene que revisar `success`.
    """

    success: bool
    content: str | None = None
    provider: str
    error: str | None = None
