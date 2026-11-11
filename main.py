"""Script de prueba: hace la misma pregunta a cada proveedor configurado,
primero en modo normal y luego en modo streaming."""

import asyncio

from dotenv import load_dotenv

from manager import PROVEEDORES, AsyncLLMManager
from schemas import ChatMessage

PREGUNTA = "¿Qué es la entropía?"


async def main() -> None:
    # Cargamos el .env antes de crear el manager, que lee las variables al iniciarse.
    load_dotenv()
    manager = AsyncLLMManager()

    messages = [
        ChatMessage(role="system", content="Eres un profesor que explica de forma breve y clara."),
        ChatMessage(role="user", content=PREGUNTA),
    ]

    for provider in PROVEEDORES:
        print(f"\n{'=' * 60}\nProveedor: {provider}\n{'=' * 60}")

        try:
            client = manager.get_client(provider)
        except ValueError as e:
            # Si falta la API key de este proveedor, avisamos y seguimos con el otro.
            print(f"[AVISO] {e}. Se omite este proveedor.")
            continue

        # 1) Modo normal: esperamos la respuesta completa.
        print(f"\n--- Modo normal ---\nPregunta: {PREGUNTA}\n")
        response = await client.generate(messages)
        if response.success:
            print(response.content)
        else:
            print(f"[ERROR] {response.error}")

        # 2) Modo streaming: imprimimos cada trozo en cuanto llega.
        print(f"\n--- Modo streaming ---\nPregunta: {PREGUNTA}\n")
        async for chunk in client.stream(messages):
            # El error llega en un campo aparte, no mezclado con el texto.
            if chunk.error:
                print(f"\n[ERROR] {chunk.error}")
                break
            # flush=True fuerza a mostrar el texto inmediatamente, sin esperar
            # a que se llene el búfer de la consola.
            print(chunk.text, end="", flush=True)
        print()


if __name__ == "__main__":
    asyncio.run(main())
