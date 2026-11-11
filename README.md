# unified-async-llm-client

Cliente asíncrono unificado para **OpenAI** y **Anthropic (Claude)**. Ofrece una
misma interfaz (`generate` y `stream`) para los dos proveedores, con validación de
datos mediante Pydantic y manejo de errores con reintentos.

## Estructura

| Archivo | Contenido |
|---|---|
| `schemas.py` | Modelos Pydantic: `ChatMessage`, `ModelConfig`, `ModelResponse`, `StreamChunk` |
| `base_client.py` | `BaseLLMClient`: interfaz común, reintentos y manejo de errores |
| `openai_client.py` | `OpenAIClient` (usa `AsyncOpenAI`) |
| `anthropic_client.py` | `AnthropicClient` (usa `AsyncAnthropic`) |
| `manager.py` | `AsyncLLMManager` y registro de proveedores: elige el cliente según el `.env` |
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
| `LLM_TEMPERATURE` | Temperatura, de 0 a 2 (opcional, por defecto `0.7`). **Solo la usa OpenAI**: los modelos actuales de Anthropic no aceptan este parámetro y se ignora | `0.7` |
| `LLM_MAX_TOKENS` | Máximo de tokens de la respuesta, de 1 a 4096 (opcional, por defecto `512`) | `512` |

Si `LLM_TEMPERATURE` o `LLM_MAX_TOKENS` tienen un valor fuera de rango, Pydantic
lo rechaza al crear el cliente y `main.py` avisa y omite ese proveedor.

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

**Lógica común en la clase base.** `BaseLLMClient` implementa `generate()` y
`stream()` con los reintentos y el manejo de errores. Cada proveedor solo
declara:

- `_request()` y `_request_stream()`: cómo llamar a su SDK.
- `transient_errors` y `auth_errors`: qué excepciones de su SDK se reintentan y
  cuáles son de autenticación.
- `provider_name` y `default_model`.

Si cambia la política de reintentos, se cambia solo en `base_client.py`.

**Manejo de errores.** Ningún error rompe el programa:

- **API key inválida** (`AuthenticationError`): se devuelve
  `ModelResponse(success=False, error=...)` sin reintentar, porque reintentar no
  lo arreglaría.
- **Rate limit o conexión/timeout** (`RateLimitError`, `APIConnectionError`):
  son errores temporales, así que se reintenta hasta 3 veces esperando 1 s, 2 s
  y 4 s con `await asyncio.sleep(...)`. Si sigue fallando, se devuelve el error.
- **Otros errores** (p. ej. modelo inexistente): se devuelven sin reintentar.
- **En streaming** `stream()` emite objetos `StreamChunk` con `text` o `error`,
  así el error nunca se confunde con el texto del modelo. Si un error temporal
  ocurre antes del primer trozo, se reintenta igual que en `generate()`. Si ya se
  emitió texto, no se reintenta, porque se repetiría la respuesta.

Los reintentos internos de los SDK se desactivan (`max_retries=0`) para que la
única lógica de reintentos sea la del proyecto.

**Agregar un proveedor.** Crea una subclase de `BaseLLMClient` (como
`openai_client.py`) y súmala a la tupla de `PROVEEDORES` en `manager.py`. El
manager lee automáticamente `<NOMBRE>_API_KEY` y `<NOMBRE>_MODEL` del `.env`.
