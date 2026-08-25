import os

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from empleo_utils import (
    ConfigError,
    ExternalServiceError,
    format_offers_markdown,
    normalize_tool_args,
    save_favorite,
    search_jobs,
)


@tool
def buscar_empleos(criterio: str):
    """
    Busca ofertas de empleo REALES en los proveedores configurados.
    Acepta términos de búsqueda sin importar mayúsculas o errores leves.
    """
    busqueda_limpia = str(criterio or "").lower().strip()

    try:
        print(f"[BUSCANDO EMPLEOS: {busqueda_limpia}...]")
        ofertas = search_jobs(busqueda_limpia)
    except ConfigError as exc:
        return str(exc)
    except ExternalServiceError as exc:
        return str(exc)

    if not ofertas:
        return f"No encontré nada para '{criterio}' en los proveedores configurados."

    return format_offers_markdown(ofertas)


@tool
def guardar_empleo(nombre_puesto: str, enlace_url: str = ""):
    """
    ÚSALO ÚNICAMENTE para guardar un empleo en el archivo de FAVORITOS.
    Extrae el nombre/empresa para el parámetro 'nombre_puesto'.
    EXTRAE OBLIGATORIAMENTE la URL (lo que empieza con https://...) y ponla en 'enlace_url'.
    """
    save_favorite(nombre_puesto, enlace_url)
    return "Guardado en favoritos"


INSTRUCCIONES = SystemMessage(content="""
Eres un asistente de empleo experto.
1. Si el usuario quiere 'guardar' un puesto, usa la herramienta 'guardar_empleo'.
2. Al guardar, DEBES copiar el texto íntegro del puesto (Título, Empresa, Lugar y LINK) que se mostró en los resultados de búsqueda. No resumas ni modifiques esa información.
3. Es vital que el link no se pierda en el proceso de guardado.
""")

HERRAMIENTAS = {
    "buscar_empleos": buscar_empleos,
    "guardar_empleo": guardar_empleo,
}


def crear_llm():
    modelo = os.getenv("OLLAMA_MODEL", "llama3.2")
    return ChatOllama(model=modelo, temperature=0).bind_tools(list(HERRAMIENTAS.values()))


def ejecutar_cli():
    llm_con_herramientas = crear_llm()
    memoria = []

    print("--- Asistente de Empleo con Poderes ---")

    while True:
        try:
            usuario = input("\n¿Qué quieres hacer? (o 'salir'): ")
        except (EOFError, KeyboardInterrupt):
            print("\nSaliendo.")
            break

        if usuario.lower().strip() == "salir":
            break

        memoria.append(HumanMessage(content=usuario))

        try:
            respuesta = llm_con_herramientas.invoke([INSTRUCCIONES] + memoria)
        except Exception as exc:
            memoria.pop()
            print(f"No pude conectar con Ollama o el modelo configurado: {exc}")
            continue

        memoria.append(respuesta)

        tool_calls = getattr(respuesta, "tool_calls", None) or []
        if not tool_calls:
            print("\nIA dice:", respuesta.content)
            continue

        for tool_call in tool_calls:
            nombre_herramienta = tool_call["name"]
            argumentos = normalize_tool_args(tool_call["args"])
            id_llamada = tool_call["id"]
            herramienta = HERRAMIENTAS.get(nombre_herramienta)

            if herramienta is None:
                resultado = f"Herramienta no disponible: {nombre_herramienta}"
            else:
                print(f"[IA EJECUTANDO: {nombre_herramienta}...]")
                try:
                    resultado = herramienta.invoke(argumentos)
                except Exception as exc:
                    resultado = f"No pude ejecutar {nombre_herramienta}: {exc}"

            print(resultado)
            memoria.append(ToolMessage(content=str(resultado), tool_call_id=id_llamada))

        try:
            segunda_respuesta = llm_con_herramientas.invoke([INSTRUCCIONES] + memoria)
        except Exception:
            continue

        memoria.append(segunda_respuesta)
        if segunda_respuesta.content:
            print("\nIA dice:", segunda_respuesta.content)


if __name__ == "__main__":
    ejecutar_cli()
