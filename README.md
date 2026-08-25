# Tiburoncín Job Bot

Asistente de empleo con dos interfaces:

- `agente.py`: CLI con Ollama y herramientas de LangChain.
- `agente_web.py`: app de Streamlit.
- `server.py`: servidor MCP para guardar postulaciones.

## Setup

```bash
python3 -m venv venv
venv/bin/pip install -r requirements.txt
venv/bin/pip install -r requirements-dev.txt
cp .env.example .env
```

Completa `.env` con al menos un proveedor de ofertas. Recomendacion para empezar:

```bash
SAMPLE_JOBS_ENABLED=true
JOB_PROVIDERS=sample,jooble,careerjet,adzuna
JOOBLE_API_KEY=...
CAREERJET_API_KEY=...
ADZUNA_APP_ID=...
ADZUNA_APP_KEY=...
OLLAMA_MODEL=llama3.2
```

Mientras no tengas APIs conectadas, `sample` carga ofertas demo desde
`ofertas.json`. Cuando conectes APIs reales, desactivalo asi:

```bash
SAMPLE_JOBS_ENABLED=false
JOB_PROVIDERS=jooble,careerjet,adzuna
```

Si quieres priorizar cobertura y datos enriquecidos de pago, pon TheirStack delante:

```bash
JOB_PROVIDERS=theirstack,jooble,careerjet,adzuna
THEIRSTACK_API_KEY=...
THEIRSTACK_COUNTRY_CODE=ES
THEIRSTACK_POSTED_AT_MAX_AGE_DAYS=30
```

No necesitas configurar todos los proveedores. El bot consultara los que tengan
credenciales y saltara los que no esten disponibles.

También necesitas tener Ollama corriendo y el modelo disponible:

```bash
ollama pull llama3.2
ollama serve
```

## Uso

CLI:

```bash
venv/bin/python agente.py
```

Streamlit:

```bash
venv/bin/streamlit run agente_web.py
```

La interfaz web incluye:

- Busqueda directa sin IA con puesto, ciudad, remoto y dias maximos.
- Selector de idioma para ver la interfaz en espanol o ingles.
- Selector de proveedores en la barra lateral.
- Selector de ciudades de Espana con campo de ciudad personalizada.
- Selector de categoria y puesto tipo, con campo libre de puesto que tiene prioridad.
- Resultados en tarjetas con enlace y boton para guardar.
- Filtros, ordenacion y comparacion sobre resultados ya cargados.
- Busquedas guardadas en `busquedas_guardadas.json`.
- Chat con Ollama usando los filtros seleccionados y respuesta en el idioma detectado del mensaje.
- Favoritos estructurados en `favoritos.json`, incluyendo ofertas guardadas desde el chat, con lectura compatible de `favoritos.txt`.
- Listado de favoritos y busquedas guardadas de mas reciente a mas antiguo.
- Estado manual por favorito: pendiente, aplicado, entrevista o descartado.
- Notas por favorito y boton para abrir oferta sin mostrar el link crudo.
- Pestaña de configuracion para probar claves API en la sesion sin editar `.env`.

Servidor MCP:

```bash
venv/bin/python server.py
```

Las postulaciones nuevas de la interfaz web se guardan en `favoritos.json`.
El archivo `favoritos.txt` se sigue leyendo para compatibilidad con datos antiguos.

## Desarrollo

```bash
venv/bin/python -m unittest discover -v
venv/bin/python -m pytest -q
venv/bin/ruff check .
```

`unittest` no requiere dependencias extra. `pytest` y `ruff` se instalan desde
`requirements-dev.txt`.
