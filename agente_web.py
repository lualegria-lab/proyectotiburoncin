from __future__ import annotations

import html
import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

from empleo_utils import (
    ConfigError,
    ExternalServiceError,
    SAMPLE_JOBS_FILE,
    clean_text,
    normalize_tool_args,
    sample_jobs_enabled,
    search_jobs,
)
from state_store import (
    FAVORITES_JSON_FILE,
    StateFileError,
    delete_favorite_record as _delete_favorite_record,
    favorite_ids as _favorite_ids,
    filter_favorite_records as _filter_favorite_records,
    format_saved_at as _format_saved_at,
    load_saved_searches as _load_saved_searches,
    newest_first as _newest_first,
    normalize_favorite_record as _normalize_favorite_record,
    now_iso as _now_iso,
    read_favorite_records as _read_favorite_records,
    save_search as _save_search,
    stable_id as _stable_id,
    update_favorite_record as _update_favorite_record,
    upsert_favorite_record as _upsert_favorite_record,
)
PROVIDER_LABELS = {
    "sample": "JSON demo",
    "jooble": "Jooble",
    "careerjet": "Careerjet",
    "adzuna": "Adzuna",
    "theirstack": "TheirStack",
}
DEFAULT_PROVIDER_VALUES = ("sample", "jooble", "careerjet", "adzuna")
SECRET_NAMES = (
    "SAMPLE_JOBS_ENABLED",
    "JOB_PROVIDERS",
    "ADZUNA_APP_ID",
    "ADZUNA_APP_KEY",
    "JOOBLE_API_KEY",
    "JOOBLE_LOCATION",
    "CAREERJET_API_KEY",
    "CAREERJET_LOCALE",
    "CAREERJET_LOCATION",
    "CAREERJET_USER_IP",
    "CAREERJET_USER_AGENT",
    "THEIRSTACK_API_KEY",
    "THEIRSTACK_COUNTRY_CODE",
    "THEIRSTACK_POSTED_AT_MAX_AGE_DAYS",
)

LANGUAGE_OPTIONS = {"Español": "es", "English": "en"}
CITY_OPTION_ALL = "__all__"
CITY_OPTION_CUSTOM = "__custom__"
SPAIN_CITIES = (
    "Madrid",
    "Barcelona",
    "Valencia",
    "Sevilla",
    "Zaragoza",
    "Malaga",
    "Murcia",
    "Palma",
    "Las Palmas de Gran Canaria",
    "Bilbao",
    "Alicante",
    "Cordoba",
    "Valladolid",
    "Vigo",
    "Gijon",
    "A Coruna",
    "Vitoria-Gasteiz",
    "Granada",
    "Elche",
    "Oviedo",
    "Santa Cruz de Tenerife",
    "Badalona",
    "Cartagena",
    "Terrassa",
    "Jerez de la Frontera",
    "Sabadell",
    "Mostoles",
    "Alcala de Henares",
    "Pamplona",
    "Almeria",
    "San Sebastian",
    "Burgos",
    "Santander",
    "Castellon de la Plana",
    "Logrono",
    "Badajoz",
    "Salamanca",
    "Huelva",
    "Lleida",
    "Tarragona",
    "Leon",
    "Cadiz",
    "Jaen",
    "Ourense",
    "Girona",
    "Lugo",
    "Caceres",
    "Toledo",
    "Guadalajara",
    "Pontevedra",
    "Palencia",
    "Ciudad Real",
    "Zamora",
    "Avila",
    "Cuenca",
    "Huesca",
    "Segovia",
    "Soria",
    "Teruel",
    "Ceuta",
    "Melilla",
)
JOB_CATALOG = {
    "technology": {
        "label": {
            "es": "Informatica y telecomunicaciones",
            "en": "IT and telecommunications",
        },
        "query": "informatica telecomunicaciones",
        "roles": [
            {"es": "Desarrollador/a backend", "en": "Backend developer", "query": "backend developer"},
            {"es": "Desarrollador/a frontend", "en": "Frontend developer", "query": "frontend developer"},
            {"es": "Desarrollador/a full stack", "en": "Full-stack developer", "query": "full stack developer"},
            {"es": "Analista de datos", "en": "Data analyst", "query": "analista datos"},
            {"es": "Ingeniero/a de datos", "en": "Data engineer", "query": "data engineer"},
            {"es": "DevOps / Cloud engineer", "en": "DevOps / Cloud engineer", "query": "devops cloud"},
            {"es": "Ciberseguridad", "en": "Cybersecurity specialist", "query": "ciberseguridad"},
            {"es": "Soporte IT / Helpdesk", "en": "IT support / Helpdesk", "query": "soporte informatico helpdesk"},
            {"es": "QA tester", "en": "QA tester", "query": "qa tester"},
            {"es": "Administrador/a de sistemas", "en": "Systems administrator", "query": "administrador sistemas"},
            {"es": "Product owner", "en": "Product owner", "query": "product owner"},
        ],
    },
    "sales": {
        "label": {"es": "Comercial y ventas", "en": "Sales"},
        "query": "comercial ventas",
        "roles": [
            {"es": "Comercial", "en": "Sales representative", "query": "comercial"},
            {"es": "Account manager", "en": "Account manager", "query": "account manager"},
            {"es": "Business development", "en": "Business development", "query": "business development"},
            {"es": "Dependiente/a", "en": "Retail sales assistant", "query": "dependiente"},
            {"es": "Televenta", "en": "Telesales", "query": "televenta"},
            {"es": "Agente inmobiliario", "en": "Real estate agent", "query": "agente inmobiliario"},
        ],
    },
    "admin": {
        "label": {
            "es": "Administracion y secretariado",
            "en": "Administration and office support",
        },
        "query": "administrativo secretaria",
        "roles": [
            {"es": "Administrativo/a", "en": "Administrative assistant", "query": "administrativo"},
            {"es": "Recepcionista", "en": "Receptionist", "query": "recepcionista"},
            {"es": "Secretario/a", "en": "Secretary", "query": "secretaria"},
            {"es": "Office manager", "en": "Office manager", "query": "office manager"},
            {"es": "Auxiliar contable", "en": "Accounting assistant", "query": "auxiliar contable"},
            {"es": "Tecnico/a de nominas", "en": "Payroll technician", "query": "tecnico nominas"},
        ],
    },
    "customer_service": {
        "label": {"es": "Atencion al cliente", "en": "Customer service"},
        "query": "atencion al cliente",
        "roles": [
            {"es": "Atencion al cliente", "en": "Customer support", "query": "atencion al cliente"},
            {"es": "Teleoperador/a", "en": "Call center agent", "query": "teleoperador"},
            {"es": "Customer success", "en": "Customer success", "query": "customer success"},
            {"es": "Soporte tecnico", "en": "Technical support", "query": "soporte tecnico"},
            {"es": "Gestor/a de incidencias", "en": "Incident manager", "query": "gestor incidencias"},
            {"es": "Postventa", "en": "After-sales support", "query": "postventa"},
        ],
    },
    "marketing": {
        "label": {"es": "Marketing y comunicacion", "en": "Marketing and communications"},
        "query": "marketing comunicacion",
        "roles": [
            {"es": "Marketing digital", "en": "Digital marketing", "query": "marketing digital"},
            {"es": "SEO / SEM specialist", "en": "SEO / SEM specialist", "query": "seo sem"},
            {"es": "Social media manager", "en": "Social media manager", "query": "social media"},
            {"es": "Content manager", "en": "Content manager", "query": "content manager"},
            {"es": "CRM marketing", "en": "CRM marketing", "query": "crm marketing"},
            {"es": "Disenador/a grafico/a", "en": "Graphic designer", "query": "diseñador grafico"},
        ],
    },
    "finance": {
        "label": {"es": "Finanzas, banca y contabilidad", "en": "Finance, banking and accounting"},
        "query": "finanzas contabilidad banca",
        "roles": [
            {"es": "Contable", "en": "Accountant", "query": "contable"},
            {"es": "Analista financiero", "en": "Financial analyst", "query": "analista financiero"},
            {"es": "Controller financiero", "en": "Financial controller", "query": "controller financiero"},
            {"es": "Auditor/a", "en": "Auditor", "query": "auditor"},
            {"es": "Asesor/a fiscal", "en": "Tax advisor", "query": "asesor fiscal"},
            {"es": "Gestor/a banca", "en": "Banking advisor", "query": "gestor banca"},
        ],
    },
    "logistics": {
        "label": {"es": "Compras, logistica y transporte", "en": "Procurement, logistics and transport"},
        "query": "logistica transporte almacen",
        "roles": [
            {"es": "Mozo/a de almacen", "en": "Warehouse operative", "query": "mozo almacen"},
            {"es": "Carretillero/a", "en": "Forklift driver", "query": "carretillero"},
            {"es": "Repartidor/a", "en": "Delivery driver", "query": "repartidor"},
            {"es": "Coordinador/a logistico/a", "en": "Logistics coordinator", "query": "coordinador logistica"},
            {"es": "Tecnico/a de compras", "en": "Procurement technician", "query": "tecnico compras"},
            {"es": "Supply chain", "en": "Supply chain", "query": "supply chain"},
        ],
    },
    "hospitality": {
        "label": {"es": "Hosteleria y turismo", "en": "Hospitality and tourism"},
        "query": "hosteleria turismo",
        "roles": [
            {"es": "Camarero/a", "en": "Waiter / waitress", "query": "camarero"},
            {"es": "Cocinero/a", "en": "Cook", "query": "cocinero"},
            {"es": "Recepcionista de hotel", "en": "Hotel receptionist", "query": "recepcionista hotel"},
            {"es": "Housekeeping", "en": "Housekeeping", "query": "housekeeping"},
            {"es": "Agente de viajes", "en": "Travel agent", "query": "agente viajes"},
            {"es": "Jefe/a de sala", "en": "Restaurant manager", "query": "jefe sala"},
        ],
    },
    "healthcare": {
        "label": {"es": "Sanidad y servicios sociales", "en": "Healthcare and social services"},
        "query": "sanidad salud servicios sociales",
        "roles": [
            {"es": "Enfermero/a", "en": "Nurse", "query": "enfermero"},
            {"es": "Auxiliar de enfermeria", "en": "Nursing assistant", "query": "auxiliar enfermeria"},
            {"es": "Medico/a", "en": "Doctor", "query": "medico"},
            {"es": "Fisioterapeuta", "en": "Physiotherapist", "query": "fisioterapeuta"},
            {"es": "Psicologo/a", "en": "Psychologist", "query": "psicologo"},
            {"es": "Farmaceutico/a", "en": "Pharmacist", "query": "farmaceutico"},
            {"es": "Trabajador/a social", "en": "Social worker", "query": "trabajador social"},
        ],
    },
    "engineering": {
        "label": {"es": "Ingenieria y produccion", "en": "Engineering and production"},
        "query": "ingenieria produccion",
        "roles": [
            {"es": "Ingeniero/a industrial", "en": "Industrial engineer", "query": "ingeniero industrial"},
            {"es": "Operario/a de produccion", "en": "Production operator", "query": "operario produccion"},
            {"es": "Tecnico/a de mantenimiento", "en": "Maintenance technician", "query": "tecnico mantenimiento"},
            {"es": "Tecnico/a de calidad", "en": "Quality technician", "query": "tecnico calidad"},
            {"es": "Ingeniero/a mecanico/a", "en": "Mechanical engineer", "query": "ingeniero mecanico"},
            {"es": "Ingeniero/a electrico/a", "en": "Electrical engineer", "query": "ingeniero electrico"},
        ],
    },
    "trades": {
        "label": {"es": "Profesiones, artes y oficios", "en": "Skilled trades and crafts"},
        "query": "oficios profesionales",
        "roles": [
            {"es": "Electricista", "en": "Electrician", "query": "electricista"},
            {"es": "Fontanero/a", "en": "Plumber", "query": "fontanero"},
            {"es": "Soldador/a", "en": "Welder", "query": "soldador"},
            {"es": "Carpintero/a", "en": "Carpenter", "query": "carpintero"},
            {"es": "Mecanico/a", "en": "Mechanic", "query": "mecanico"},
            {"es": "Personal de limpieza", "en": "Cleaner", "query": "limpieza"},
            {"es": "Jardinero/a", "en": "Gardener", "query": "jardinero"},
        ],
    },
    "education": {
        "label": {"es": "Educacion y formacion", "en": "Education and training"},
        "query": "educacion formacion",
        "roles": [
            {"es": "Profesor/a", "en": "Teacher", "query": "profesor"},
            {"es": "Profesor/a de idiomas", "en": "Language teacher", "query": "profesor idiomas"},
            {"es": "Formador/a", "en": "Trainer", "query": "formador"},
            {"es": "Monitor/a", "en": "Activity monitor", "query": "monitor"},
            {"es": "Pedagogo/a", "en": "Pedagogue", "query": "pedagogo"},
        ],
    },
    "hr": {
        "label": {"es": "Recursos humanos", "en": "Human resources"},
        "query": "recursos humanos",
        "roles": [
            {"es": "Recruiter", "en": "Recruiter", "query": "recruiter"},
            {"es": "Tecnico/a de RRHH", "en": "HR specialist", "query": "tecnico recursos humanos"},
            {"es": "Relaciones laborales", "en": "Labor relations", "query": "relaciones laborales"},
            {"es": "Tecnico/a de seleccion", "en": "Talent acquisition", "query": "seleccion personal"},
            {"es": "Tecnico/a de formacion", "en": "Training specialist", "query": "tecnico formacion"},
        ],
    },
    "legal": {
        "label": {"es": "Legal", "en": "Legal"},
        "query": "legal abogado",
        "roles": [
            {"es": "Abogado/a", "en": "Lawyer", "query": "abogado"},
            {"es": "Asesor/a legal", "en": "Legal advisor", "query": "asesor legal"},
            {"es": "Paralegal", "en": "Paralegal", "query": "paralegal"},
            {"es": "Compliance officer", "en": "Compliance officer", "query": "compliance officer"},
        ],
    },
    "construction": {
        "label": {"es": "Construccion e inmobiliaria", "en": "Construction and real estate"},
        "query": "construccion inmobiliaria",
        "roles": [
            {"es": "Arquitecto/a", "en": "Architect", "query": "arquitecto"},
            {"es": "Jefe/a de obra", "en": "Site manager", "query": "jefe obra"},
            {"es": "Peon de construccion", "en": "Construction laborer", "query": "peon construccion"},
            {"es": "Aparejador/a", "en": "Quantity surveyor", "query": "aparejador"},
            {"es": "Ingeniero/a de caminos", "en": "Civil engineer", "query": "ingeniero caminos"},
        ],
    },
}
TRANSLATIONS = {
    "es": {
        "app_title": "¿Con instinto de tiburón?",
        "app_subtitle": "¡Vamos a por tu próxima oportunidad!",
        "app_shark_emoji": "🦈",
        "language": "Idioma / Language",
        "sources": "Fuentes",
        "providers": "Proveedores",
        "filters": "Filtros",
        "default_city": "Ciudad por defecto",
        "default_city_custom": "Ciudad personalizada por defecto",
        "default_remote": "Solo remoto por defecto",
        "max_days": "Dias maximos",
        "results_per_source": "Resultados por fuente",
        "all_spain": "Toda Espana",
        "custom_option": "Personalizado",
        "city_selector": "Ciudad",
        "custom_city": "Ciudad personalizada",
        "custom_city_placeholder": "Ej. Alcobendas, remoto Espana...",
        "category": "Categoria",
        "all_categories": "Todas las categorias",
        "role_type": "Puesto tipo",
        "any_role": "Cualquier puesto de la categoria",
        "custom_role": "Puesto personalizado",
        "custom_role_placeholder": "Ej. Prompt engineer, analista GIS...",
        "status": "Estado",
        "configured": "configurado",
        "missing_key": "sin clave",
        "favorites": "Favoritos",
        "favorites_empty": "No hay favoritos guardados",
        "favorites_hint": "El estado se actualiza manualmente desde aqui.",
        "favorite_search": "Buscar favoritos",
        "favorite_search_placeholder": "Puesto, empresa, ciudad, nota...",
        "filtered_favorites_empty": "No hay favoritos con esos filtros.",
        "open_job": "Abrir oferta",
        "save": "Guardar",
        "already_saved": "Ya guardado",
        "saved": "Guardado en favoritos",
        "saved_at": "Guardado",
        "details": "Detalle",
        "status_pending": "Pendiente",
        "status_applied": "Aplicado",
        "status_interview": "Entrevista",
        "status_discarded": "Archivado",
        "status_filter": "Estado",
        "all_statuses": "Todos los estados",
        "mark_pending": "Pendiente",
        "mark_applied": "Aplicado",
        "mark_interview": "Entrevista",
        "mark_discarded": "Archivar",
        "delete_favorite": "Borrar",
        "confirm_delete": "Confirmar borrado",
        "cancel": "Cancelar",
        "favorite_deleted": "Favorito borrado",
        "notes": "Notas",
        "save_note": "Guardar nota",
        "note_saved": "Nota guardada",
        "results_filters": "Filtros sobre resultados",
        "text_filter": "Texto",
        "source_filter": "Fuente",
        "city_filter": "Ciudad",
        "sort_by": "Ordenar por",
        "sort_relevance": "Relevancia",
        "sort_title": "Puesto",
        "sort_company": "Empresa",
        "sort_city": "Ciudad",
        "sort_source": "Fuente",
        "compare_select": "Comparar",
        "comparison": "Comparacion",
        "saved_searches": "Busquedas guardadas",
        "save_search": "Guardar busqueda",
        "search_saved": "Busqueda guardada",
        "run_search": "Ejecutar",
        "no_saved_searches": "No hay busquedas guardadas",
        "config": "Configuracion",
        "temp_api_keys": "Claves temporales de API",
        "temp_api_hint": "Se guardan solo en esta sesion de Streamlit.",
        "save_config": "Guardar configuracion",
        "config_saved": "Configuracion guardada",
        "sample_jobs": "Usar JSON demo",
        "no_current_results": "No hay resultados para los filtros actuales.",
        "results": "Resultados",
        "no_results": "No encontre ofertas con esos filtros. Prueba con otro puesto, otra ciudad o ampliando los dias de busqueda.",
        "tool_saved": "Guardado en favoritos",
        "tool_result": "Resultado de herramienta",
        "chat_recent_results": "Ofertas recientes del chat",
        "chat_results_ready": "He encontrado {count} ofertas. Puedes abrirlas o guardarlas desde las tarjetas.",
        "chat_saved_offer": "Guardado en favoritos",
        "chat_no_recent_results": "No hay ofertas recientes del chat para guardar.",
        "chat_offer_not_found": "No encontre esa oferta entre los ultimos resultados del chat.",
        "chat_input": "Buscar, comparar o guardar ofertas",
        "chat_submit": "Enviar",
        "chat_thinking": "Pensando...",
        "chat_shortcut_remote": "Busca empleos remotos",
        "chat_shortcut_recent": "Busca ofertas recientes",
        "chat_shortcut_save_first": "Guarda la primera oferta",
        "chat_clear": "Limpiar chat",
        "ollama_error": "No pude conectar con Ollama o el modelo configurado",
        "ollama_complete_error": "No pude completar la respuesta con Ollama",
        "tool_unavailable": "Herramienta no disponible",
        "chat_action_error": "No pude completar esa accion",
        "job_title": "Puesto",
        "city": "Ciudad",
        "remote_only": "Solo remoto",
        "days": "Dias",
        "direct_search": "Buscar sin IA",
        "enter_job": "Introduce un puesto para buscar.",
        "select_source": "Selecciona al menos una fuente.",
        "searching": "Buscando ofertas...",
        "refresh": "Actualizar",
        "tab_search": "Buscar ofertas",
        "tab_chat": "Chat con Tiburoncín",
        "tab_favorites": "Favoritos",
        "tab_config": "Configuracion",
        "job_na": "Puesto N/A",
        "company_na": "Empresa N/A",
        "source_na": "Fuente N/A",
        "at_word": "en",
        "chat_instructions": """Eres un asistente de empleo experto.
1. Entiende español e ingles.
2. Responde en el idioma del ultimo mensaje del usuario.
3. Cuando busques, usa la busqueda interna.
4. Muestra resultados con numero, titulo, empresa, lugar y fuente.
5. No muestres URLs ni enlaces Markdown en la respuesta: la interfaz mostrara los botones Abrir oferta y Guardar.
6. Si el usuario quiere guardar una oferta ya mostrada, no busques otra vez: guarda la oferta usando su numero, titulo, empresa o enlace.
7. No inventes ofertas ni enlaces.""",
    },
    "en": {
        "app_title": "Feeling sharky?",
        "app_subtitle": "Let’s hunt down your next opportunity!",
        "app_shark_emoji": "🦈",
        "language": "Idioma / Language",
        "sources": "Sources",
        "providers": "Providers",
        "filters": "Filters",
        "default_city": "Default city",
        "default_city_custom": "Custom default city",
        "default_remote": "Remote only by default",
        "max_days": "Max days",
        "results_per_source": "Results per source",
        "all_spain": "All Spain",
        "custom_option": "Custom",
        "city_selector": "City",
        "custom_city": "Custom city",
        "custom_city_placeholder": "E.g. Alcobendas, remote Spain...",
        "category": "Category",
        "all_categories": "All categories",
        "role_type": "Typical role",
        "any_role": "Any role in this category",
        "custom_role": "Custom role",
        "custom_role_placeholder": "E.g. Prompt engineer, GIS analyst...",
        "status": "Status",
        "configured": "configured",
        "missing_key": "missing key",
        "favorites": "Favorites",
        "favorites_empty": "No favorites saved",
        "favorites_hint": "Status is updated manually from here.",
        "favorite_search": "Search favorites",
        "favorite_search_placeholder": "Role, company, city, note...",
        "filtered_favorites_empty": "No favorites match those filters.",
        "open_job": "Open job",
        "save": "Save",
        "already_saved": "Already saved",
        "saved": "Saved to favorites",
        "saved_at": "Saved",
        "details": "Details",
        "status_pending": "Pending",
        "status_applied": "Applied",
        "status_interview": "Interview",
        "status_discarded": "Archived",
        "status_filter": "Status",
        "all_statuses": "All statuses",
        "mark_pending": "Pending",
        "mark_applied": "Applied",
        "mark_interview": "Interview",
        "mark_discarded": "Archive",
        "delete_favorite": "Delete",
        "confirm_delete": "Confirm delete",
        "cancel": "Cancel",
        "favorite_deleted": "Favorite deleted",
        "notes": "Notes",
        "save_note": "Save note",
        "note_saved": "Note saved",
        "results_filters": "Result filters",
        "text_filter": "Text",
        "source_filter": "Source",
        "city_filter": "City",
        "sort_by": "Sort by",
        "sort_relevance": "Relevance",
        "sort_title": "Role",
        "sort_company": "Company",
        "sort_city": "City",
        "sort_source": "Source",
        "compare_select": "Compare",
        "comparison": "Comparison",
        "saved_searches": "Saved searches",
        "save_search": "Save search",
        "search_saved": "Search saved",
        "run_search": "Run",
        "no_saved_searches": "No saved searches",
        "config": "Settings",
        "temp_api_keys": "Temporary API keys",
        "temp_api_hint": "Stored only in this Streamlit session.",
        "save_config": "Save settings",
        "config_saved": "Settings saved",
        "sample_jobs": "Use JSON demo",
        "no_current_results": "No results for the current filters.",
        "results": "Results",
        "no_results": "I could not find jobs with those filters. Try another role, city, or a wider date range.",
        "tool_saved": "Saved to favorites",
        "tool_result": "Tool result",
        "chat_recent_results": "Recent chat jobs",
        "chat_results_ready": "I found {count} jobs. You can open or save them from the cards.",
        "chat_saved_offer": "Saved to favorites",
        "chat_no_recent_results": "There are no recent chat jobs to save.",
        "chat_offer_not_found": "I could not find that job in the latest chat results.",
        "chat_input": "Search, compare, or save jobs",
        "chat_submit": "Send",
        "chat_thinking": "Thinking...",
        "chat_shortcut_remote": "Find remote jobs",
        "chat_shortcut_recent": "Find recent jobs",
        "chat_shortcut_save_first": "Save the first job",
        "chat_clear": "Clear chat",
        "ollama_error": "Could not connect to Ollama or the configured model",
        "ollama_complete_error": "Could not complete the response with Ollama",
        "tool_unavailable": "Tool unavailable",
        "chat_action_error": "I could not complete that action",
        "job_title": "Role",
        "city": "City",
        "remote_only": "Remote only",
        "days": "Days",
        "direct_search": "Search without AI",
        "enter_job": "Enter a role to search.",
        "select_source": "Select at least one source.",
        "searching": "Searching jobs...",
        "refresh": "Refresh",
        "tab_search": "Search jobs",
        "tab_chat": "Chat with Sharky",
        "tab_favorites": "Favorites",
        "tab_config": "Settings",
        "job_na": "Role N/A",
        "company_na": "Company N/A",
        "source_na": "Source N/A",
        "at_word": "at",
        "chat_instructions": """You are an expert job-search assistant.
1. Understand Spanish and English.
2. Respond in the language of the user's latest message.
3. When searching, use the internal search.
4. Show results with number, title, company, location, and source.
5. Do not show URLs or Markdown links in the response: the interface will render Open job and Save buttons.
6. If the user wants to save a job already shown, do not search again: save the job using its number, title, company, or link.
7. Do not invent jobs or links.""",
    },
}


def _language_code() -> str:
    return st.session_state.get("language", "es")


def _translation(key: str, language: str | None = None) -> str:
    language = language or _language_code()
    return TRANSLATIONS.get(language, TRANSLATIONS["es"]).get(
        key, TRANSLATIONS["es"].get(key, key)
    )


def _t(key: str) -> str:
    return _translation(key)


def _label(key: str, language: str | None = None) -> str:
    return _translation(key, language) if language else _t(key)


def _detect_chat_language(text: str, fallback_language: str | None = None) -> str:
    normalized = str(text or "").lower()
    tokens = set(re.findall(r"[a-záéíóúüñ']+", normalized, flags=re.IGNORECASE))
    spanish_terms = {
        "busca",
        "buscar",
        "buscame",
        "búscame",
        "quiero",
        "necesito",
        "empleo",
        "empleos",
        "trabajo",
        "trabajos",
        "oferta",
        "ofertas",
        "puesto",
        "puestos",
        "guarda",
        "guardar",
        "guardame",
        "guárdame",
        "favorito",
        "favoritos",
        "remoto",
        "remota",
        "ciudad",
        "empresa",
        "salario",
        "sueldo",
        "este",
        "esta",
        "segundo",
        "segunda",
        "primero",
        "primera",
        "proxima",
        "próxima",
    }
    english_terms = {
        "find",
        "search",
        "show",
        "save",
        "job",
        "jobs",
        "role",
        "roles",
        "offer",
        "offers",
        "favorite",
        "favorites",
        "remote",
        "city",
        "company",
        "salary",
        "this",
        "that",
        "first",
        "second",
        "next",
        "opportunity",
        "apply",
        "interview",
    }
    spanish_score = sum(term in tokens for term in spanish_terms)
    english_score = sum(term in tokens for term in english_terms)

    if re.search(r"[¿¡áéíóúüñ]", normalized):
        spanish_score += 2
    if re.search(r"\b(let's|dont|don't|can't|can you|please)\b", normalized):
        english_score += 2

    if english_score > spanish_score:
        return "en"
    if spanish_score > english_score:
        return "es"
    if fallback_language in TRANSLATIONS:
        return fallback_language
    return _language_code() if _language_code() in TRANSLATIONS else "es"


def _chat_language_code() -> str:
    language = st.session_state.get("chat_response_language", _language_code())
    return language if language in TRANSLATIONS else _language_code()


def _chat_t(key: str) -> str:
    return _translation(key, _chat_language_code())


def _chat_system_message(language: str) -> SystemMessage:
    response_language = "English" if language == "en" else "Spanish"
    return SystemMessage(
        content=(
            "You are an expert job-search assistant.\n"
            "You understand Spanish and English.\n"
            f"The latest user message is in {response_language}. "
            f"Respond only in {response_language} for this turn.\n"
            "When searching, use the internal job search tool.\n"
            "Show results with number, title, company, location, and source.\n"
            "Do not show URLs or Markdown links in the response: "
            "the interface renders Open job and Save buttons for each result.\n"
            "If the user wants to save a job already shown, do not search again: "
            "save the job using its number, title, company, or link.\n"
            "Do not invent jobs or links."
        )
    )


def _status_options() -> tuple[str, ...]:
    return ("pending", "applied", "interview", "discarded")


def _status_label(status: str) -> str:
    normalized = status if status in _status_options() else "pending"
    return _t(f"status_{normalized}")


def _city_options() -> list[str]:
    return [CITY_OPTION_ALL, *SPAIN_CITIES, CITY_OPTION_CUSTOM]


def _city_label(option: str) -> str:
    if option == CITY_OPTION_ALL:
        return _t("all_spain")
    if option == CITY_OPTION_CUSTOM:
        return _t("custom_option")
    return option


def _city_option_for_location(location: str) -> str:
    cleaned_location = str(location or "").strip()
    if not cleaned_location:
        return CITY_OPTION_ALL
    if cleaned_location in SPAIN_CITIES:
        return cleaned_location
    return CITY_OPTION_CUSTOM


def _resolve_city(option: str, custom_city: str = "") -> str:
    cleaned_custom_city = str(custom_city or "").strip()
    if cleaned_custom_city:
        return cleaned_custom_city
    if option in (CITY_OPTION_ALL, CITY_OPTION_CUSTOM):
        return ""
    return option


def _category_options() -> list[str]:
    return ["all", *JOB_CATALOG.keys()]


def _category_label(category_key: str) -> str:
    if category_key == "all":
        return _t("all_categories")
    category = JOB_CATALOG[category_key]
    return category["label"][_language_code()]


def _role_options(category_key: str) -> list[str]:
    if category_key == "all":
        return ["any"]
    return ["any", *[str(index) for index in range(len(JOB_CATALOG[category_key]["roles"]))]]


def _role_label(category_key: str, role_key: str) -> str:
    if role_key == "any":
        return _t("any_role")
    role = JOB_CATALOG[category_key]["roles"][int(role_key)]
    return role[_language_code()]


def _resolve_role_query(
    category_key: str,
    role_key: str,
    custom_role: str,
) -> str:
    cleaned_custom_role = str(custom_role or "").strip()
    if cleaned_custom_role:
        return cleaned_custom_role
    if category_key == "all":
        return ""
    if role_key == "any":
        return str(JOB_CATALOG[category_key]["query"])
    return str(JOB_CATALOG[category_key]["roles"][int(role_key)]["query"])


def _get_secret(name: str) -> str | None:
    try:
        value = st.secrets.get(name)
    except Exception:
        return None

    return str(value).strip() if value else None


def _get_config_value(name: str) -> str:
    overrides = st.session_state.get("api_overrides", {})
    if name in overrides:
        return str(overrides.get(name) or "").strip()
    return (_get_secret(name) or os.getenv(name) or "").strip()


def _get_job_credentials() -> dict[str, str | None]:
    return {name: _get_config_value(name) or None for name in SECRET_NAMES}


def _configured_provider_values() -> list[str]:
    configured = _get_config_value("JOB_PROVIDERS")
    values = configured.split(",") if configured else DEFAULT_PROVIDER_VALUES
    valid_values = []

    for value in values:
        normalized = value.strip().lower()
        if normalized in PROVIDER_LABELS and normalized not in valid_values:
            valid_values.append(normalized)

    if sample_jobs_enabled(credentials=_get_job_credentials()) and "sample" not in valid_values:
        valid_values.insert(0, "sample")

    return valid_values or list(DEFAULT_PROVIDER_VALUES)


def _provider_is_configured(provider: str) -> bool:
    if provider == "sample":
        return sample_jobs_enabled(credentials=_get_job_credentials()) and SAMPLE_JOBS_FILE.exists()
    if provider == "jooble":
        return bool(_get_config_value("JOOBLE_API_KEY"))
    if provider == "careerjet":
        return bool(_get_config_value("CAREERJET_API_KEY"))
    if provider == "adzuna":
        return bool(_get_config_value("ADZUNA_APP_ID")) and bool(
            _get_config_value("ADZUNA_APP_KEY")
        )
    if provider == "theirstack":
        return bool(_get_config_value("THEIRSTACK_API_KEY"))
    return False


def _offer_value(offer: Mapping[str, Any], *keys: str, default: str = "") -> str:
    current: Any = offer
    for key in keys:
        if not isinstance(current, Mapping):
            return default
        current = current.get(key)
    return clean_text(current, default)


def _offer_detail(
    offer: Mapping[str, Any],
    *,
    language: str | None = None,
) -> tuple[str, str, str, str, str, str]:
    title = clean_text(offer.get("title"), _label("job_na", language))
    company = _offer_value(offer, "company", "display_name", default=_label("company_na", language))
    location = _offer_value(offer, "location", "display_name", default="N/A")
    provider = clean_text(offer.get("_provider"), _label("source_na", language))
    salary = clean_text(offer.get("_salary"), "")
    link = clean_text(offer.get("redirect_url"), "")
    return title, company, location, provider, salary, link


def _favorite_text(offer: Mapping[str, Any]) -> str:
    title, company, location, provider, salary, _link = _offer_detail(offer)
    salary_part = f" - {salary}" if salary else ""
    description = clean_text(offer.get("description"), "")
    description_part = f" - {description[:260]}" if description else ""
    return (
        f"{title} {_t('at_word')} {company} ({location}) [{provider}]"
        f"{salary_part}{description_part}"
    )


def _offer_search_text(offer: Mapping[str, Any]) -> str:
    title, company, location, provider, salary, link = _offer_detail(offer)
    description = clean_text(offer.get("description"), "")
    return f"{title} {company} {location} {provider} {salary} {description} {link}".lower()


def _extract_first_url(text: str) -> str:
    match = re.search(r"https?://[^\s)>\]]+", str(text or ""))
    return match.group(0).rstrip(".,;") if match else ""


def _reference_index(reference: str, offer_count: int) -> int | None:
    normalized = str(reference or "").lower()
    number_match = re.search(r"(?:^|\D)(\d{1,2})(?:\D|$)", normalized)
    if number_match:
        index = int(number_match.group(1)) - 1
        if 0 <= index < offer_count:
            return index

    word_indexes = {
        "first": 0,
        "1st": 0,
        "primero": 0,
        "primera": 0,
        "segundo": 1,
        "segunda": 1,
        "second": 1,
        "2nd": 1,
        "tercero": 2,
        "tercera": 2,
        "third": 2,
        "3rd": 2,
        "cuarto": 3,
        "cuarta": 3,
        "fourth": 3,
        "4th": 3,
        "quinto": 4,
        "quinta": 4,
        "fifth": 4,
        "5th": 4,
    }
    for word, index in word_indexes.items():
        if re.search(rf"\b{re.escape(word)}\b", normalized) and index < offer_count:
            return index

    if re.search(r"\b(last|ultimo|ultima|último|última)\b", normalized):
        return offer_count - 1 if offer_count else None

    return None


def _looks_like_chat_offer_reference(reference: str) -> bool:
    normalized = str(reference or "").lower().strip()
    if not normalized:
        return False
    return bool(
        re.search(
            r"\b(\d{1,2}|first|second|third|fourth|fifth|last|primero|primera|segundo|segunda|tercero|tercera|cuarto|cuarta|quinto|quinta|ultimo|ultima|último|última|este|esta|ese|esa|alguno|alguna|cualquiera|that|it|one|them)\b",
            normalized,
        )
    )


def _resolve_chat_offer_reference(
    reference: str,
    link: str = "",
    *,
    offers: Sequence[Mapping[str, Any]] | None = None,
) -> Mapping[str, Any] | None:
    chat_offers = list(offers if offers is not None else st.session_state.get("chat_search_results", []))
    if not chat_offers:
        return None

    resolved_link = clean_text(link, "") or _extract_first_url(reference)
    if resolved_link:
        for offer in chat_offers:
            if clean_text(offer.get("redirect_url"), "") == resolved_link:
                return offer

    reference_text = clean_text(reference, "").lower()
    index = _reference_index(reference_text, len(chat_offers))
    if index is not None:
        return chat_offers[index]

    if reference_text in {"it", "that", "that one", "este", "esta", "ese", "esa"} and len(chat_offers) == 1:
        return chat_offers[0]

    meaningful_terms = [
        term
        for term in re.findall(r"[\wáéíóúüñ]+", reference_text, flags=re.IGNORECASE)
        if len(term) > 2
        and term
        not in {
            "guardar",
            "guarda",
            "save",
            "favoritos",
            "favorites",
            "oferta",
            "job",
            "empleo",
            "puesto",
            "the",
            "that",
            "them",
            "one",
            "este",
            "esta",
            "ese",
            "esa",
            "alguno",
            "alguna",
            "cualquiera",
        }
    ]
    if not meaningful_terms:
        return None

    matches = [
        offer
        for offer in chat_offers
        if all(term in _offer_search_text(offer) for term in meaningful_terms)
    ]
    if len(matches) == 1:
        return matches[0]

    partial_matches = [
        offer
        for offer in chat_offers
        if any(term in _offer_search_text(offer) for term in meaningful_terms)
    ]
    return partial_matches[0] if len(partial_matches) == 1 else None


def _favorite_record_from_offer(offer: Mapping[str, Any]) -> dict[str, str]:
    title, company, location, provider, salary, link = _offer_detail(offer)
    description = clean_text(offer.get("description"), "")
    return _normalize_favorite_record(
        {
            "id": _stable_id(title, company, location, link),
            "title": title,
            "company": company,
            "location": location,
            "provider": provider,
            "salary": salary,
            "description": description,
            "link": link,
            "status": "pending",
            "saved_at": _now_iso(),
        }
    )


def _favorite_id_from_offer(offer: Mapping[str, Any]) -> str:
    title, company, location, _provider, _salary, link = _offer_detail(offer)
    return _stable_id(title, company, location, link)


def _manual_favorite_record(detail: str, link: str = "") -> dict[str, str]:
    cleaned_detail = clean_text(detail, _t("favorites"))
    cleaned_link = clean_text(link, "")
    return _normalize_favorite_record(
        {
            "id": _stable_id(cleaned_detail, cleaned_link),
            "title": cleaned_detail,
            "link": cleaned_link,
            "status": "pending",
            "saved_at": _now_iso(),
        }
    )


def _save_offer_to_favorites(
    offer: Mapping[str, Any],
    *,
    file_path: Path = FAVORITES_JSON_FILE,
) -> dict[str, str]:
    return _upsert_favorite_record(
        _favorite_record_from_offer(offer),
        file_path=file_path,
        default_title=_t("favorites"),
    )


def _favorite_ids_for_ui() -> set[str]:
    return _favorite_ids(_read_favorite_records(default_title=_t("favorites")))


def _filter_values(favorites: Sequence[Mapping[str, str]], field: str) -> list[str]:
    return sorted(
        {
            clean_text(favorite.get(field), "")
            for favorite in favorites
            if clean_text(favorite.get(field), "")
        }
    )


def _save_chat_offer_reference(
    reference: str,
    link: str = "",
    *,
    file_path: Path = FAVORITES_JSON_FILE,
    language: str | None = None,
) -> str | None:
    recent_offers = st.session_state.get("chat_search_results", [])
    if not recent_offers:
        return _label("chat_no_recent_results", language) if _looks_like_chat_offer_reference(reference) else None

    offer = _resolve_chat_offer_reference(reference, link, offers=recent_offers)
    if offer is None:
        return _label("chat_offer_not_found", language) if _looks_like_chat_offer_reference(reference) or link else None

    _save_offer_to_favorites(offer, file_path=file_path)
    title = _offer_detail(offer, language=language)[0]
    return f"{_label('chat_saved_offer', language)}: {title}"


def _render_status_badge(status: str) -> None:
    normalized = status if status in _status_options() else "pending"
    st.markdown(
        f"<span class='status-badge status-{normalized}'>{html.escape(_status_label(normalized))}</span>",
        unsafe_allow_html=True,
    )


def _render_favorite_summary(
    favorite: Mapping[str, str],
    *,
    key_prefix: str,
    index: int,
    show_management: bool = True,
) -> None:
    title = clean_text(favorite.get("title"), _t("favorites"))
    company = clean_text(favorite.get("company"), "")
    location = clean_text(favorite.get("location"), "")
    provider = clean_text(favorite.get("provider"), "")
    salary = clean_text(favorite.get("salary"), "")
    description = clean_text(favorite.get("description"), "")
    notes = clean_text(favorite.get("notes"), "")
    status = clean_text(favorite.get("status"), "pending")
    link = clean_text(favorite.get("link"), "")
    saved_at = _format_saved_at(favorite.get("saved_at"))

    _render_status_badge(status)
    st.markdown(
        f"<div class='favorite-title'>{html.escape(title)}</div>",
        unsafe_allow_html=True,
    )

    saved_at_label = f"{_t('saved_at')}: {saved_at}" if saved_at else ""
    meta_items = [item for item in (saved_at_label, company, location, provider, salary) if item]
    if meta_items:
        st.markdown(
            "<div class='offer-meta'>"
            + "".join(f"<span>{html.escape(item)}</span>" for item in meta_items)
            + "</div>",
            unsafe_allow_html=True,
        )

    if description:
        st.markdown(
            f"<div class='offer-description'>{html.escape(description[:420])}</div>",
            unsafe_allow_html=True,
        )

    if link:
        st.link_button(_t("open_job"), link, use_container_width=True)

    if not show_management:
        return

    status_cols = st.columns(4)
    for col, target_status, label_key in zip(
        status_cols,
        _status_options(),
        ("mark_pending", "mark_applied", "mark_interview", "mark_discarded"),
    ):
        disabled = status == target_status
        if col.button(
            _t(label_key),
            key=f"{key_prefix}_status_{target_status}_{favorite['id']}_{index}",
            disabled=disabled,
            use_container_width=True,
        ):
            try:
                _update_favorite_record(
                    favorite,
                    status=target_status,
                    default_title=_t("favorites"),
                )
            except StateFileError as exc:
                st.error(str(exc))
            else:
                st.rerun()

    with st.expander(_t("notes")):
        note_value = st.text_area(
            _t("notes"),
            value=notes,
            key=f"{key_prefix}_notes_{favorite['id']}_{index}",
            label_visibility="collapsed",
        )
        if st.button(_t("save_note"), key=f"{key_prefix}_save_note_{favorite['id']}_{index}"):
            try:
                _update_favorite_record(
                    favorite,
                    notes=note_value,
                    default_title=_t("favorites"),
                )
            except StateFileError as exc:
                st.error(str(exc))
            else:
                st.success(_t("note_saved"))

    delete_state_key = f"{key_prefix}_confirm_delete_{favorite['id']}_{index}"
    if st.session_state.get(delete_state_key):
        delete_cols = st.columns([0.5, 0.5])
        if delete_cols[0].button(
            _t("confirm_delete"),
            key=f"{key_prefix}_delete_confirmed_{favorite['id']}_{index}",
            use_container_width=True,
        ):
            try:
                _delete_favorite_record(
                    favorite["id"],
                    default_title=_t("favorites"),
                )
            except StateFileError as exc:
                st.error(str(exc))
            else:
                st.session_state.pop(delete_state_key, None)
                st.session_state.favorite_flash = _t("favorite_deleted")
                st.rerun()
        if delete_cols[1].button(
            _t("cancel"),
            key=f"{key_prefix}_delete_cancel_{favorite['id']}_{index}",
            use_container_width=True,
        ):
            st.session_state.pop(delete_state_key, None)
            st.rerun()
    elif st.button(
        _t("delete_favorite"),
        key=f"{key_prefix}_delete_{favorite['id']}_{index}",
        use_container_width=True,
    ):
        st.session_state[delete_state_key] = True
        st.rerun()


def _render_provider_status(selected_providers: list[str]) -> None:
    st.sidebar.markdown(f"### {_t('status')}")
    for provider in selected_providers:
        label = PROVIDER_LABELS[provider]
        state = _t("configured") if _provider_is_configured(provider) else _t("missing_key")
        state_class = "ok" if _provider_is_configured(provider) else "warn"
        st.sidebar.markdown(
            f"<div class='provider-status {state_class}'>"
            f"<span>{html.escape(label)}</span><strong>{state}</strong></div>",
            unsafe_allow_html=True,
        )


def _render_favorites_preview() -> None:
    try:
        favorites = [
            favorite
            for favorite in _read_favorite_records(default_title=_t("favorites"))
            if favorite.get("status", "pending") != "discarded"
        ]
    except StateFileError as exc:
        st.sidebar.error(str(exc))
        return

    st.sidebar.markdown(f"### {_t('favorites')}")
    if not favorites:
        st.sidebar.caption(_t("favorites_empty"))
        return

    for index, favorite in enumerate(favorites[:4]):
        with st.sidebar.container(border=True):
            _render_favorite_summary(
                favorite,
                key_prefix="sidebar_favorite",
                index=index,
                show_management=False,
            )


def _run_search(
    query: str,
    *,
    providers: list[str],
    location: str,
    remote: bool,
    max_age_days: int,
    results_per_provider: int,
) -> tuple[list[dict[str, Any]], str | None]:
    try:
        offers = search_jobs(
            query,
            providers=providers,
            credentials=_get_job_credentials(),
            location=location,
            remote=remote,
            posted_at_max_age_days=max_age_days,
            results_per_provider=results_per_provider,
        )
    except ConfigError as exc:
        return [], str(exc)
    except ExternalServiceError as exc:
        return [], str(exc)

    return offers, None


def _render_offer_card(
    offer: Mapping[str, Any],
    index: int,
    key_prefix: str,
    favorite_ids: set[str] | None = None,
    *,
    display_index: int | None = None,
) -> None:
    title, company, location, provider, salary, link = _offer_detail(offer)
    description = clean_text(offer.get("description"), "")
    updated = clean_text(offer.get("_updated"), "")
    favorite_id = _favorite_id_from_offer(offer)
    is_saved = favorite_id in (favorite_ids or set())
    display_title = f"{display_index}. {title}" if display_index is not None else title

    with st.container(border=True):
        top_left, top_right = st.columns([0.76, 0.24])
        with top_left:
            st.markdown(
                f"<div class='offer-provider'>{html.escape(provider)}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(f"#### {html.escape(display_title)}")
            st.markdown(
                "<div class='offer-meta'>"
                f"<span>{html.escape(company)}</span>"
                f"<span>{html.escape(location)}</span>"
                f"{f'<span>{html.escape(salary)}</span>' if salary else ''}"
                f"{f'<span>{html.escape(updated)}</span>' if updated else ''}"
                "</div>",
                unsafe_allow_html=True,
            )
        with top_right:
            if link and link != "#":
                st.link_button(_t("open_job"), link, use_container_width=True)
            if is_saved:
                st.button(
                    _t("already_saved"),
                    key=f"{key_prefix}_saved_{favorite_id}_{index}",
                    disabled=True,
                    use_container_width=True,
                )
            elif st.button(_t("save"), key=f"{key_prefix}_save_{index}", use_container_width=True):
                try:
                    saved_record = _save_offer_to_favorites(offer)
                except StateFileError as exc:
                    st.error(str(exc))
                else:
                    if favorite_ids is not None:
                        favorite_ids.add(saved_record["id"])
                    st.success(_t("saved"))

        if description:
            st.markdown(
                f"<div class='offer-description'>{html.escape(description[:360])}</div>",
                unsafe_allow_html=True,
            )
            with st.expander(_t("details")):
                st.write(description)


def _render_chat_offer_actions(
    offers: Sequence[Mapping[str, Any]],
    *,
    key_prefix: str,
    favorite_ids: set[str] | None = None,
) -> None:
    if not offers:
        return

    st.markdown(f"### {_t('chat_recent_results')} ({len(offers)})")
    for index, offer in enumerate(offers):
        _render_offer_card(
            offer,
            index,
            key_prefix,
            favorite_ids,
            display_index=index + 1,
        )


def _filtered_sorted_offers(
    offers: list[dict[str, Any]],
    *,
    key_prefix: str,
) -> list[dict[str, Any]]:
    with st.expander(_t("results_filters"), expanded=False):
        filter_cols = st.columns([0.32, 0.24, 0.24, 0.20])
        text_filter = filter_cols[0].text_input(
            _t("text_filter"), key=f"{key_prefix}_text_filter"
        ).lower().strip()
        provider_options = sorted(
            {
                clean_text(offer.get("_provider"), "")
                for offer in offers
                if clean_text(offer.get("_provider"), "")
            }
        )
        city_options = sorted(
            {
                _offer_value(offer, "location", "display_name", default="")
                for offer in offers
                if _offer_value(offer, "location", "display_name", default="")
            }
        )
        selected_providers = filter_cols[1].multiselect(
            _t("source_filter"),
            options=provider_options,
            key=f"{key_prefix}_provider_filter",
        )
        selected_cities = filter_cols[2].multiselect(
            _t("city_filter"),
            options=city_options,
            key=f"{key_prefix}_city_filter",
        )
        sort_options = {
            _t("sort_relevance"): "relevance",
            _t("sort_title"): "title",
            _t("sort_company"): "company",
            _t("sort_city"): "city",
            _t("sort_source"): "source",
        }
        sort_label = filter_cols[3].selectbox(
            _t("sort_by"),
            options=list(sort_options.keys()),
            key=f"{key_prefix}_sort",
        )

    filtered = []
    for offer in offers:
        title, company, location, provider, salary, _link = _offer_detail(offer)
        description = clean_text(offer.get("description"), "")
        searchable = f"{title} {company} {location} {provider} {salary} {description}".lower()

        if text_filter and text_filter not in searchable:
            continue
        if selected_providers and provider not in selected_providers:
            continue
        if selected_cities and location not in selected_cities:
            continue
        filtered.append(offer)

    sort_key = sort_options[sort_label]
    if sort_key != "relevance":
        key_map = {
            "title": lambda offer: _offer_detail(offer)[0].lower(),
            "company": lambda offer: _offer_detail(offer)[1].lower(),
            "city": lambda offer: _offer_detail(offer)[2].lower(),
            "source": lambda offer: _offer_detail(offer)[3].lower(),
        }
        filtered.sort(key=key_map[sort_key])

    return filtered


def _render_comparison(offers: list[dict[str, Any]], *, key_prefix: str) -> None:
    selected = []
    for index, offer in enumerate(offers):
        title = _offer_detail(offer)[0]
        if st.checkbox(
            f"{_t('compare_select')}: {title}",
            key=f"{key_prefix}_compare_{index}",
        ):
            selected.append(offer)

    if len(selected) < 2:
        return

    st.markdown(f"### {_t('comparison')}")
    columns = st.columns(min(len(selected), 3))
    for column, offer in zip(columns, selected[:3]):
        title, company, location, provider, salary, link = _offer_detail(offer)
        description = clean_text(offer.get("description"), "")
        with column:
            st.markdown(f"#### {html.escape(title)}")
            st.write(company)
            st.write(location)
            st.write(provider)
            if salary:
                st.write(salary)
            if description:
                st.caption(description[:220])
            if link:
                st.link_button(_t("open_job"), link, use_container_width=True)


def _render_saved_searches() -> None:
    with st.expander(_t("saved_searches"), expanded=False):
        try:
            searches = _load_saved_searches()
        except StateFileError as exc:
            st.error(str(exc))
            return

        if not searches:
            st.caption(_t("no_saved_searches"))
        for index, search in enumerate(_newest_first(searches)):
            query = clean_text(search.get("query"), "")
            location = clean_text(search.get("location"), _t("all_spain"))
            providers = ", ".join(search.get("providers", []))
            cols = st.columns([0.72, 0.28])
            cols[0].markdown(
                f"**{html.escape(query)}** · {html.escape(location)} · {html.escape(providers)}"
            )
            if cols[1].button(_t("run_search"), key=f"run_saved_search_{index}", use_container_width=True):
                with st.spinner(_t("searching")):
                    offers, error = _run_search(
                        query,
                        providers=list(search.get("providers", DEFAULT_PROVIDER_VALUES)),
                        location=str(search.get("location", "")),
                        remote=bool(search.get("remote", False)),
                        max_age_days=int(search.get("max_age_days", 30)),
                        results_per_provider=int(search.get("results_per_provider", 5)),
                    )
                if error:
                    st.error(error)
                else:
                    st.session_state.direct_search_results = offers
                    st.session_state.direct_search_query = query
                    st.rerun()


def _render_results(offers: list[dict[str, Any]], *, key_prefix: str) -> None:
    if not offers:
        st.info(_t("no_current_results"))
        return

    visible_offers = _filtered_sorted_offers(offers, key_prefix=key_prefix)
    st.markdown(f"### {_t('results')} ({len(visible_offers)})")
    _render_comparison(visible_offers, key_prefix=key_prefix)
    try:
        saved_favorite_ids = _favorite_ids_for_ui()
    except StateFileError as exc:
        saved_favorite_ids = set()
        st.warning(str(exc))
    for index, offer in enumerate(visible_offers):
        _render_offer_card(offer, index, key_prefix, saved_favorite_ids)


def _format_offers_markdown(
    offers: list[dict[str, Any]],
    *,
    language: str | None = None,
) -> str:
    lines = []
    for index, offer in enumerate(offers, start=1):
        title, company, location, provider, salary, _link = _offer_detail(offer, language=language)
        provider_prefix = f"[{provider}] " if provider else ""
        salary_suffix = f" - {salary}" if salary else ""
        lines.append(
            f"{index}. {provider_prefix}{title} {_label('at_word', language)} {company} ({location})"
            f"{salary_suffix}."
        )

    return "\n".join(lines)


def _ensure_chat_state() -> None:
    if "llm" not in st.session_state:
        model = _get_secret("OLLAMA_MODEL") or os.getenv("OLLAMA_MODEL", "llama3.2")
        st.session_state.llm = ChatOllama(model=model, temperature=0).bind_tools(
            [buscar_empleos, guardar_empleo]
        )

    if "memoria" not in st.session_state:
        st.session_state.memoria = []

    st.session_state.instrucciones = _chat_system_message(_language_code())


def _search_context() -> dict[str, Any]:
    return st.session_state.get(
        "search_context",
        {
            "providers": list(DEFAULT_PROVIDER_VALUES),
            "location": "",
            "remote": False,
            "max_age_days": 30,
            "results_per_provider": 5,
        },
    )


@tool
def buscar_empleos(criterio: str):
    """Busca empleos en los proveedores configurados."""
    context = _search_context()
    offers, error = _run_search(
        criterio,
        providers=context["providers"],
        location=context["location"],
        remote=context["remote"],
        max_age_days=context["max_age_days"],
        results_per_provider=context["results_per_provider"],
    )
    if error:
        return error
    if not offers:
        st.session_state.chat_search_results = []
        st.session_state.chat_search_query = criterio
        return _chat_t("no_results")
    st.session_state.chat_search_results = [dict(offer) for offer in offers]
    st.session_state.chat_search_query = criterio
    return _format_offers_markdown(offers, language=_chat_language_code())


@tool
def guardar_empleo(detalle_completo: str, enlace_url: str = ""):
    """Guarda el detalle de un empleo en favoritos.json."""
    try:
        chat_result = _save_chat_offer_reference(
            detalle_completo,
            enlace_url,
            language=_chat_language_code(),
        )
        if chat_result:
            return chat_result
        resolved_link = clean_text(enlace_url, "") or _extract_first_url(detalle_completo)
        _upsert_favorite_record(
            _manual_favorite_record(detalle_completo, resolved_link),
            default_title=_t("favorites"),
        )
    except StateFileError as exc:
        return str(exc)
    return _chat_t("tool_saved")


def _is_internal_tool_content(content: Any) -> bool:
    text = str(content or "").strip()
    if not text:
        return False
    internal_markers = (
        "buscar_empleos",
        '"buscar_empleos"',
        "'buscar_empleos'",
        "guardar_empleo",
        '"guardar_empleo"',
        "'guardar_empleo'",
        "tool_calls",
        '"tool_calls"',
        "'tool_calls'",
        '"parameters"',
        "'parameters'",
    )
    return any(marker in text for marker in internal_markers)


def _strip_chat_links(content: str) -> str:
    link_labels = "Abrir oferta|Open job|Enlace|Link"
    text = re.sub(
        rf"\s*\.?\s*\[(?:{link_labels})\]\(https?://[^)]+\)",
        ".",
        str(content or ""),
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\s+-\s*(?:Enlace|Link):\s*https?://\S+",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\[([^\]]+)\]\(https?://[^)]+\)", r"\1", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"[ \t]+([.,;:])", r"\1", text)
    text = re.sub(r"\.{2,}", ".", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _visible_ai_content(message: AIMessage) -> str:
    if getattr(message, "tool_calls", None):
        return ""
    content = str(message.content or "").strip()
    if _is_internal_tool_content(content):
        return ""
    return _strip_chat_links(content)


def _render_chat_history() -> None:
    for msg in reversed(st.session_state.memoria):
        if isinstance(msg, HumanMessage):
            st.chat_message("user").write(msg.content)
        elif isinstance(msg, AIMessage):
            content = _visible_ai_content(msg)
            if content:
                st.chat_message("assistant").write(content)


def _run_chat_tool_call(tool_call: Mapping[str, Any]) -> str:
    args = normalize_tool_args(tool_call["args"])
    tool_name = tool_call["name"]
    tool_map = {
        "buscar_empleos": buscar_empleos,
        "guardar_empleo": guardar_empleo,
    }
    tool_func = tool_map.get(tool_name)
    if tool_func is None:
        return _chat_t("tool_unavailable")

    try:
        return str(tool_func.invoke(args))
    except Exception as exc:
        return f"{_chat_t('chat_action_error')}: {exc}"


def _visible_tool_result(tool_name: str, result: str, language: str) -> str:
    if tool_name == "buscar_empleos" and st.session_state.get("chat_search_results"):
        count = len(st.session_state.get("chat_search_results", []))
        return _translation("chat_results_ready", language).format(count=count)
    return _strip_chat_links(result)


def _handle_chat_prompt(prompt: str) -> None:
    response_language = _detect_chat_language(
        prompt,
        fallback_language=st.session_state.get("chat_response_language"),
    )
    st.session_state.chat_response_language = response_language
    instructions = _chat_system_message(response_language)
    st.session_state.memoria.append(HumanMessage(content=prompt))
    context_messages = [instructions] + st.session_state.memoria

    try:
        response = st.session_state.llm.invoke(context_messages)
    except Exception as exc:
        st.session_state.memoria.append(
            AIMessage(content=f"{_translation('ollama_error', response_language)}: {exc}")
        )
        return

    st.session_state.memoria.append(response)
    tool_calls = getattr(response, "tool_calls", None) or []
    if not tool_calls:
        return

    tool_results = []
    with st.spinner(_t("searching")):
        for tool_call in tool_calls:
            result = _run_chat_tool_call(tool_call)
            tool_results.append(
                _visible_tool_result(tool_call["name"], result, response_language)
            )
            st.session_state.memoria.append(
                ToolMessage(content=result, tool_call_id=tool_call["id"])
            )

        st.session_state.memoria.append(
            AIMessage(content=_strip_chat_links("\n\n".join(tool_results)))
        )


def _render_chat_tab() -> None:
    _ensure_chat_state()

    shortcut_prompt = ""
    shortcut_cols = st.columns([0.26, 0.26, 0.30, 0.18])
    if shortcut_cols[0].button(
        _t("chat_shortcut_remote"),
        key="chat_shortcut_remote",
        use_container_width=True,
    ):
        shortcut_prompt = _t("chat_shortcut_remote")
    if shortcut_cols[1].button(
        _t("chat_shortcut_recent"),
        key="chat_shortcut_recent",
        use_container_width=True,
    ):
        shortcut_prompt = _t("chat_shortcut_recent")
    if shortcut_cols[2].button(
        _t("chat_shortcut_save_first"),
        key="chat_shortcut_save_first",
        disabled=not bool(st.session_state.get("chat_search_results")),
        use_container_width=True,
    ):
        shortcut_prompt = _t("chat_shortcut_save_first")
    if shortcut_cols[3].button(
        _t("chat_clear"),
        key="chat_clear",
        disabled=not bool(
            st.session_state.get("memoria")
            or st.session_state.get("chat_search_results")
        ),
        use_container_width=True,
    ):
        st.session_state.memoria = []
        st.session_state.chat_search_results = []
        st.session_state.chat_search_query = ""
        st.session_state.chat_response_language = _language_code()

    with st.form("chat_prompt_form", clear_on_submit=True):
        prompt = st.text_input(
            _t("chat_input"),
            placeholder=_t("chat_input"),
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button(_t("chat_submit"), use_container_width=True)

    active_prompt = shortcut_prompt or (prompt.strip() if submitted else "")
    if active_prompt:
        with st.spinner(_t("chat_thinking")):
            _handle_chat_prompt(active_prompt)

    try:
        saved_favorite_ids = _favorite_ids_for_ui()
    except StateFileError as exc:
        saved_favorite_ids = set()
        st.warning(str(exc))

    _render_chat_offer_actions(
        st.session_state.get("chat_search_results", []),
        key_prefix="chat_cached",
        favorite_ids=saved_favorite_ids,
    )

    _render_chat_history()


def _render_search_tab(
    selected_providers: list[str],
    sidebar_location: str,
    sidebar_remote: bool,
    sidebar_max_age: int,
    results_per_provider: int,
) -> None:
    with st.container(border=True):
        first_row = st.columns([0.34, 0.36, 0.30])
        category_key = first_row[0].selectbox(
            _t("category"),
            options=_category_options(),
            format_func=_category_label,
            key="direct_category",
        )
        role_key = first_row[1].selectbox(
            _t("role_type"),
            options=_role_options(category_key),
            format_func=lambda value: _role_label(category_key, value),
            key=f"direct_role_{category_key}",
        )
        custom_role = first_row[2].text_input(
            _t("custom_role"),
            placeholder=_t("custom_role_placeholder"),
            key="direct_custom_role",
        )

        city_default_option = _city_option_for_location(sidebar_location)
        city_default_index = _city_options().index(city_default_option)
        custom_city_default = (
            sidebar_location if city_default_option == CITY_OPTION_CUSTOM else ""
        )
        second_row = st.columns([0.34, 0.34, 0.16, 0.16])
        city_option = second_row[0].selectbox(
            _t("city_selector"),
            options=_city_options(),
            index=city_default_index,
            format_func=_city_label,
            key="direct_city",
        )
        custom_city = second_row[1].text_input(
            _t("custom_city"),
            value=custom_city_default,
            placeholder=_t("custom_city_placeholder"),
            key="direct_custom_city",
        )
        remote = second_row[2].checkbox(_t("remote_only"), value=sidebar_remote)
        max_age_days = second_row[3].number_input(
            _t("days"),
            min_value=1,
            max_value=90,
            value=sidebar_max_age,
            step=1,
        )
        submitted = st.button(_t("direct_search"), use_container_width=True)

    if submitted:
        query = _resolve_role_query(category_key, role_key, custom_role)
        location = _resolve_city(city_option, custom_city)
        if not query:
            st.warning(_t("enter_job"))
        elif not selected_providers:
            st.warning(_t("select_source"))
        else:
            with st.spinner(_t("searching")):
                offers, error = _run_search(
                    query,
                    providers=selected_providers,
                    location=location,
                    remote=remote,
                    max_age_days=int(max_age_days),
                    results_per_provider=results_per_provider,
                )

            if error:
                st.error(error)
            else:
                st.session_state.direct_search_results = offers
                st.session_state.direct_search_query = query
                st.session_state.last_search = {
                    "query": query,
                    "location": location,
                    "remote": remote,
                    "max_age_days": int(max_age_days),
                    "results_per_provider": results_per_provider,
                    "providers": selected_providers,
                }

    last_search = st.session_state.get("last_search")
    if last_search:
        save_cols = st.columns([0.78, 0.22])
        save_cols[0].caption(
            f"{last_search['query']} · {last_search.get('location') or _t('all_spain')}"
        )
        if save_cols[1].button(_t("save_search"), use_container_width=True):
            try:
                _save_search(last_search)
            except StateFileError as exc:
                st.error(str(exc))
            else:
                st.success(_t("search_saved"))

    _render_saved_searches()
    _render_results(st.session_state.get("direct_search_results", []), key_prefix="direct")


def _render_favorites_tab() -> None:
    try:
        favorites = _read_favorite_records(default_title=_t("favorites"))
    except StateFileError as exc:
        st.error(str(exc))
        return

    top_left, top_right = st.columns([0.8, 0.2])
    top_left.markdown(f"### {_t('favorites')} ({len(favorites)})")
    if top_right.button(_t("refresh"), use_container_width=True):
        st.rerun()
    if flash_message := st.session_state.pop("favorite_flash", ""):
        st.success(flash_message)

    if not favorites:
        st.info(_t("favorites_empty"))
        return

    st.caption(_t("favorites_hint"))
    filter_cols = st.columns([0.34, 0.22, 0.22, 0.22])
    text_filter = filter_cols[0].text_input(
        _t("favorite_search"),
        placeholder=_t("favorite_search_placeholder"),
        key="favorite_text_filter",
    )
    status_labels = [_t("all_statuses"), *[_status_label(status) for status in _status_options()]]
    selected_status_label = filter_cols[1].selectbox(
        _t("status_filter"),
        status_labels,
        key="favorite_status_filter",
    )
    selected_status = ""
    if selected_status_label != _t("all_statuses"):
        selected_status = next(
            status for status in _status_options() if _status_label(status) == selected_status_label
        )
    selected_providers = filter_cols[2].multiselect(
        _t("source_filter"),
        options=_filter_values(favorites, "provider"),
        key="favorite_provider_filter",
    )
    selected_locations = filter_cols[3].multiselect(
        _t("city_filter"),
        options=_filter_values(favorites, "location"),
        key="favorite_location_filter",
    )

    visible_favorites = _filter_favorite_records(
        favorites,
        text=text_filter,
        status=selected_status,
        providers=selected_providers,
        locations=selected_locations,
    )
    st.caption(f"{len(visible_favorites)} / {len(favorites)}")
    if not visible_favorites:
        st.info(_t("filtered_favorites_empty"))
        return

    for index, favorite in enumerate(visible_favorites):
        with st.container(border=True):
            _render_favorite_summary(favorite, key_prefix="favorite", index=index)


def _render_config_tab() -> None:
    st.markdown(f"### {_t('temp_api_keys')}")
    st.caption(_t("temp_api_hint"))

    if "api_overrides" not in st.session_state:
        st.session_state.api_overrides = {}

    with st.form("api_config_form"):
        sample_enabled = st.checkbox(
            _t("sample_jobs"),
            value=sample_jobs_enabled(credentials=_get_job_credentials()),
        )
        jooble_key = st.text_input("JOOBLE_API_KEY", value=_get_config_value("JOOBLE_API_KEY"), type="password")
        careerjet_key = st.text_input("CAREERJET_API_KEY", value=_get_config_value("CAREERJET_API_KEY"), type="password")
        adzuna_id = st.text_input("ADZUNA_APP_ID", value=_get_config_value("ADZUNA_APP_ID"))
        adzuna_key = st.text_input("ADZUNA_APP_KEY", value=_get_config_value("ADZUNA_APP_KEY"), type="password")
        theirstack_key = st.text_input("THEIRSTACK_API_KEY", value=_get_config_value("THEIRSTACK_API_KEY"), type="password")
        submitted = st.form_submit_button(_t("save_config"), use_container_width=True)

    if submitted:
        st.session_state.api_overrides = {
            "SAMPLE_JOBS_ENABLED": "true" if sample_enabled else "false",
            "JOOBLE_API_KEY": jooble_key,
            "CAREERJET_API_KEY": careerjet_key,
            "ADZUNA_APP_ID": adzuna_id,
            "ADZUNA_APP_KEY": adzuna_key,
            "THEIRSTACK_API_KEY": theirstack_key,
        }
        st.success(_t("config_saved"))
        st.rerun()


def _apply_theme() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 1180px;
            padding-top: 1.25rem;
            padding-bottom: 2.5rem;
        }
        h1, h2, h3, h4 {
            letter-spacing: 0;
        }
        h1 {
            font-size: 2rem;
            margin-bottom: 0.4rem;
        }
        .app-subtitle {
            color: var(--text-color);
            font-size: 1.25rem;
            font-weight: 600;
            line-height: 1.35;
            margin: -0.1rem 0 1.35rem;
        }
        .app-shark {
            display: inline-block;
            font-size: 1.25em;
            margin-left: 0.35rem;
            vertical-align: -0.08em;
        }
        h4 {
            font-size: 1.05rem;
        }
        :root {
            --jobbot-sidebar-background: var(--secondary-background-color, #111827);
        }
        @supports (background-color: color-mix(in srgb, #000000 90%, #ffffff 10%)) {
            :root {
                --jobbot-sidebar-background: color-mix(
                    in srgb,
                    var(--background-color, #111827) 88%,
                    var(--text-color, #ffffff) 12%
                );
            }
        }
        [data-testid="stSidebar"],
        [data-testid="stSidebar"] > div,
        [data-testid="stSidebarContent"],
        [data-testid="stSidebarUserContent"] {
            background-color: var(--jobbot-sidebar-background) !important;
            background-image: none !important;
        }
        [data-testid="stSidebar"] {
            border-right: 1px solid rgba(128, 128, 128, 0.22);
            color: var(--text-color);
        }
        [data-testid="stSidebarCollapsedControl"] {
            background-color: var(--jobbot-sidebar-background) !important;
            border-radius: 8px;
        }
        @media (max-width: 768px) {
            [data-testid="stSidebar"],
            [data-testid="stSidebar"] > div,
            [data-testid="stSidebarContent"],
            [data-testid="stSidebarUserContent"] {
                background-color: var(--jobbot-sidebar-background) !important;
                opacity: 1 !important;
            }
            [data-testid="stSidebar"] {
                box-shadow: 0.35rem 0 1.6rem rgba(0, 0, 0, 0.35);
            }
        }
        [data-testid="stSidebar"] h3 {
            margin-top: 0.35rem;
            color: var(--text-color);
        }
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] p {
            color: var(--text-color);
        }
        [data-testid="stSidebar"] input {
            color: var(--text-color);
        }
        div[data-testid="stForm"],
        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 8px;
        }
        div[data-testid="stButton"] button,
        div[data-testid="stLinkButton"] a,
        div[data-testid="stFormSubmitButton"] button {
            border-radius: 8px;
            min-height: 2.3rem;
            font-weight: 600;
        }
        .provider-status {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 0.75rem;
            padding: 0.45rem 0.6rem;
            margin: 0.3rem 0;
            border: 1px solid rgba(128, 128, 128, 0.24);
            border-radius: 8px;
            background: var(--background-color);
            color: var(--text-color);
            font-size: 0.86rem;
        }
        .provider-status strong {
            font-size: 0.78rem;
            text-transform: uppercase;
        }
        .provider-status.ok strong {
            color: #047857;
        }
        .provider-status.warn strong {
            color: #b45309;
        }
        .favorite-preview {
            padding: 0.55rem 0.65rem;
            margin: 0.35rem 0;
            border: 1px solid rgba(128, 128, 128, 0.24);
            border-radius: 8px;
            background: var(--background-color);
            color: var(--text-color);
            font-size: 0.83rem;
            line-height: 1.35;
        }
        .favorite-title {
            color: var(--text-color);
            font-size: 0.94rem;
            font-weight: 600;
            line-height: 1.4;
            margin-bottom: 0.55rem;
            overflow-wrap: anywhere;
        }
        .status-badge {
            display: inline-flex;
            align-items: center;
            padding: 0.14rem 0.5rem;
            border-radius: 999px;
            font-size: 0.76rem;
            font-weight: 800;
            margin-bottom: 0.45rem;
            border: 1px solid rgba(128, 128, 128, 0.24);
        }
        .status-pending {
            background: #fff7ed;
            color: #9a3412;
        }
        .status-applied {
            background: #ecfdf5;
            color: #047857;
        }
        .status-interview {
            background: #eff6ff;
            color: #1d4ed8;
        }
        .status-discarded {
            background: #f3f4f6;
            color: #4b5563;
        }
        .offer-provider {
            display: inline-flex;
            align-items: center;
            padding: 0.12rem 0.45rem;
            border-radius: 999px;
            background: #e8f3ff;
            color: #075985;
            font-size: 0.78rem;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }
        .offer-meta {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin: 0.2rem 0 0.15rem;
        }
        .offer-meta span {
            padding: 0.18rem 0.5rem;
            border: 1px solid rgba(128, 128, 128, 0.24);
            border-radius: 999px;
            color: var(--text-color);
            background: var(--secondary-background-color);
            font-size: 0.82rem;
        }
        .offer-description {
            margin-top: 0.65rem;
            color: var(--text-color);
            line-height: 1.45;
            font-size: 0.92rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="¿Con instinto de tiburón?", page_icon="T", layout="wide")
    _apply_theme()

    language_labels = list(LANGUAGE_OPTIONS.keys())
    current_language = st.session_state.get("language", "es")
    default_language_label = next(
        label for label, code in LANGUAGE_OPTIONS.items() if code == current_language
    )
    selected_language_label = st.sidebar.selectbox(
        "Idioma / Language",
        options=language_labels,
        index=language_labels.index(default_language_label),
    )
    st.session_state.language = LANGUAGE_OPTIONS[selected_language_label]

    st.title(_t("app_title"))
    st.markdown(
        "<div class='app-subtitle'>"
        f"{html.escape(_t('app_subtitle'))}"
        f"<span class='app-shark'>{html.escape(_t('app_shark_emoji'))}</span>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.sidebar.markdown(f"### {_t('sources')}")
    default_provider_values = _configured_provider_values()
    default_provider_labels = [PROVIDER_LABELS[value] for value in default_provider_values]
    selected_provider_labels = st.sidebar.multiselect(
        _t("providers"),
        options=list(PROVIDER_LABELS.values()),
        default=default_provider_labels,
    )
    selected_provider_values = [
        value for value, label in PROVIDER_LABELS.items() if label in selected_provider_labels
    ]

    st.sidebar.markdown(f"### {_t('filters')}")
    sidebar_city_option = st.sidebar.selectbox(
        _t("default_city"),
        options=_city_options(),
        format_func=_city_label,
    )
    sidebar_custom_city = st.sidebar.text_input(
        _t("default_city_custom"), value="", placeholder=_t("custom_city_placeholder")
    )
    sidebar_location = _resolve_city(sidebar_city_option, sidebar_custom_city)
    sidebar_remote = st.sidebar.checkbox(_t("default_remote"), value=False)
    sidebar_max_age = st.sidebar.slider(_t("max_days"), min_value=1, max_value=90, value=30)
    results_per_provider = st.sidebar.number_input(
        _t("results_per_source"), min_value=1, max_value=20, value=5, step=1
    )

    st.session_state.search_context = {
        "providers": selected_provider_values,
        "location": sidebar_location,
        "remote": sidebar_remote,
        "max_age_days": int(sidebar_max_age),
        "results_per_provider": int(results_per_provider),
    }

    _render_provider_status(selected_provider_values)
    _render_favorites_preview()

    tab_search, tab_chat, tab_favorites, tab_config = st.tabs(
        [_t("tab_search"), _t("tab_chat"), _t("tab_favorites"), _t("tab_config")]
    )

    with tab_search:
        _render_search_tab(
            selected_provider_values,
            sidebar_location,
            sidebar_remote,
            int(sidebar_max_age),
            int(results_per_provider),
        )

    with tab_chat:
        _render_chat_tab()

    with tab_favorites:
        _render_favorites_tab()

    with tab_config:
        _render_config_tab()


if __name__ == "__main__":
    main()
