from mcp.server.fastmcp import FastMCP

from empleo_utils import save_favorite

# Creamos el servidor (el empleado)
mcp = FastMCP("MiAsistenteDeEmpleo")

# Esta es la "Herramienta" que el empleado sabe usar
@mcp.tool()
def anotar_postulacion(nombre_puesto: str):
    """
    Esta herramienta sirve para GUARDAR un empleo en la lista de favoritos.
    Se activa cuando el usuario dice 'guardame este' o 'me interesa este'.
    """
    save_favorite(nombre_puesto)
    
    return f"¡Listo! Anoté '{nombre_puesto}' en tu archivo de favoritos."

if __name__ == "__main__":
    mcp.run()
