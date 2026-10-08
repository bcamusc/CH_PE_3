"""
Demo end-to-end del sistema RAG.

  1) Prueba 1 — pregunta cuya respuesta SÍ está en los documentos.
  2) Prueba 2 — "pregunta trampa": respuesta que NO está (el modelo no debe alucinar).
  3) Modo interactivo por consola.

Uso:
    python main.py          # pruebas + modo interactivo
    python main.py --test   # solo las pruebas automáticas
"""

import asyncio
import sys

from rag import RAGResponse, get_rag_response

PREGUNTA_CON_RESPUESTA = (
    "¿Cuántos días de vacaciones corresponden a un empleado con 5 años de antigüedad?"
)
PREGUNTA_TRAMPA = "¿Cuál es la política de bonos por rendimiento anual en TechCorp?"


def imprimir_resultado(resultado: RAGResponse) -> None:
    print(f"🤖 Respuesta: {resultado.respuesta}")
    print(f"📎 Fuentes: {', '.join(resultado.fuentes) if resultado.fuentes else '—'}")
    print(f"🔢 Fragmentos usados: {resultado.fragmentos_recuperados}")


async def correr_pruebas() -> None:
    print("=" * 80)
    print("🧪 PRUEBA 1 — la respuesta SÍ está en los documentos")
    print("=" * 80)
    print(f"❓ Pregunta: {PREGUNTA_CON_RESPUESTA}\n")
    imprimir_resultado(await get_rag_response(PREGUNTA_CON_RESPUESTA))

    print()
    print("=" * 80)
    print("🧪 PRUEBA 2 — pregunta trampa: la respuesta NO está en los documentos")
    print("=" * 80)
    print(f"❓ Pregunta: {PREGUNTA_TRAMPA}\n")
    imprimir_resultado(await get_rag_response(PREGUNTA_TRAMPA))
    print()


async def modo_interactivo() -> None:
    print("=" * 80)
    print("💬 Modo interactivo — pregunta sobre las políticas de TechCorp")
    print("   (escribe 'salir' para terminar)")
    print("=" * 80)

    while True:
        pregunta = input("\n🧑 Tú: ").strip()
        if pregunta.lower() in ("salir", "exit", "quit", ""):
            print("\n👋 Listo, terminamos la sesión.")
            break

        resultado = await get_rag_response(pregunta)
        print()
        imprimir_resultado(resultado)


async def main() -> None:
    await correr_pruebas()
    if "--test" not in sys.argv:
        await modo_interactivo()


if __name__ == "__main__":
    asyncio.run(main())
