# unified-async-llm-client

Cliente asíncrono unificado para **OpenAI** y **Anthropic (Claude)**. Ofrece una
misma interfaz (`generate` y `stream`) para los dos proveedores, con validación de
datos mediante Pydantic y manejo de errores con reintentos.

## Estructura

| Archivo | Contenido |
|---|---|
| `schemas.py` | Modelos Pydantic: `ChatMessage`, `ModelConfig`, `ModelResponse` |
| `base_client.py` | `BaseLLMClient`, la interfaz abstracta común |
| `openai_client.py` | `OpenAIClient` (usa `AsyncOpenAI`) |
| `anthropic_client.py` | `AnthropicClient` (usa `AsyncAnthropic`) |
| `manager.py` | `AsyncLLMManager`: elige el cliente según el `.env` |
| `main.py` | Script de prueba |

## Instalación

Requiere **Python 3.12**.

```bash
# Crear y activar el entorno virtual
python -m venv .venv          

# Instalar dependencias
pip install openai anthropic pydantic python-dotenv
```

## Configuración

Copia el archivo de ejemplo y rellena tus claves:

```bash
cp .env.example .env              # en Windows: copy .env.example .env
```

| Variable | Descripción | Ejemplo |
|---|---|---|
| `LLM_PROVIDER` | Proveedor por defecto de `get_client()` | `openai` o `anthropic` |
| `OPENAI_API_KEY` | API key de OpenAI | `sk-...` |
| `ANTHROPIC_API_KEY` | API key de Anthropic | `sk-ant-...` |
| `OPENAI_MODEL` | Modelo a utilizar de OpenAI | `gpt-4o-mini` |
| `ANTHROPIC_MODEL` | Modelo a utilizar de Anthropic | `claude-sonnet-5-5` |

## Ejecución

```bash
python main.py
```

Para cada proveedor con API key configurada, el script pregunta
*"¿Qué es la entropía?"* en modo normal y en modo streaming. Si falta una
clave, avisa y continúa con el otro proveedor.

## Decisiones de diseño

**¿Por qué async?** Una llamada a un LLM pasa casi todo el tiempo esperando la
respuesta de la red. Con `async`/`await` el programa puede atender otras tareas
mientras espera (por ejemplo, varias peticiones a la vez con `asyncio.gather`).
Por eso se usan solo `AsyncOpenAI` y `AsyncAnthropic`: un cliente síncrono dentro
de una función `async` bloquearía el event loop entero.

**Manejo de errores.** Ningún error rompe el programa:

- **API key inválida** (`AuthenticationError`): se devuelve
  `ModelResponse(success=False, error=...)` sin reintentar, porque reintentar no
  lo arreglaría.
- **Rate limit o conexión/timeout** (`RateLimitError`, `APIConnectionError`):
  son errores temporales, así que se reintenta hasta 3 veces esperando 1 s, 2 s
  y 4 s con `await asyncio.sleep(...)`. Si sigue fallando, se devuelve el error.
- **Otros errores de la API** (`APIStatusError`, p. ej. modelo inexistente): se
  devuelven sin reintentar.
- **En streaming** el error se emite como un último texto `"\n[ERROR] ..."`.

Los reintentos internos de los SDK se desactivan (`max_retries=0`) para que la
única lógica de reintentos sea la del proyecto.
