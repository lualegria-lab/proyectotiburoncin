from __future__ import annotations

import html
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import requests

try:
    from dotenv import load_dotenv
except ImportError:
    pass
else:
    load_dotenv()


PROJECT_ROOT = Path(__file__).resolve().parent
FAVORITES_FILE = PROJECT_ROOT / "favoritos.txt"
SAMPLE_JOBS_FILE = PROJECT_ROOT / "ofertas.json"
ADZUNA_URL = "https://api.adzuna.com/v1/api/jobs/es/search/1"
JOOBLE_URL_TEMPLATE = "https://jooble.org/api/{api_key}"
CAREERJET_URL = "https://search.api.careerjet.net/v4/query"
THEIRSTACK_URL = "https://api.theirstack.com/v1/jobs/search"
DEFAULT_TIMEOUT_SECONDS = 10
DEFAULT_RESULTS_PER_PROVIDER = 5
DEFAULT_JOB_PROVIDERS = ("sample", "jooble", "careerjet", "adzuna")
SUPPORTED_JOB_PROVIDERS = ("sample", "theirstack", "jooble", "careerjet", "adzuna")

_TAG_RE = re.compile(r"<[^>]+>")


class ConfigError(RuntimeError):
    """Raised when required local configuration is missing."""


class ExternalServiceError(RuntimeError):
    """Raised when an external service cannot return usable data."""


def search_jobs(
    criterio: str,
    *,
    providers: Sequence[str] | None = None,
    credentials: Mapping[str, str | None] | None = None,
    location: str | None = None,
    remote: bool = False,
    posted_at_max_age_days: int | None = None,
    results_per_provider: int = DEFAULT_RESULTS_PER_PROVIDER,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> list[dict[str, Any]]:
    query = str(criterio or "").lower().strip()
    if not query:
        return []

    provider_names = resolve_job_providers(providers, credentials=credentials)
    configured_provider_count = 0
    successful_provider_count = 0
    external_errors: list[str] = []
    offers: list[dict[str, Any]] = []

    searchers: dict[str, Callable[[], list[dict[str, Any]]]] = {
        "sample": lambda: search_sample_jobs(
            query,
            credentials=credentials,
            location=location,
            remote=remote,
            results_per_page=results_per_provider,
        ),
        "adzuna": lambda: search_adzuna(
            query,
            credentials=credentials,
            location=location,
            remote=remote,
            posted_at_max_age_days=posted_at_max_age_days,
            results_per_page=results_per_provider,
            timeout=timeout,
        ),
        "jooble": lambda: search_jooble(
            query,
            credentials=credentials,
            location=location,
            remote=remote,
            results_per_page=results_per_provider,
            timeout=timeout,
        ),
        "careerjet": lambda: search_careerjet(
            query,
            credentials=credentials,
            location=location,
            remote=remote,
            results_per_page=results_per_provider,
            timeout=timeout,
        ),
        "theirstack": lambda: search_theirstack(
            query,
            credentials=credentials,
            location=location,
            remote=remote,
            posted_at_max_age_days=posted_at_max_age_days,
            results_per_page=results_per_provider,
            timeout=timeout,
        ),
    }

    for provider_name in provider_names:
        try:
            provider_offers = searchers[provider_name]()
        except ConfigError:
            continue
        except ExternalServiceError as exc:
            configured_provider_count += 1
            external_errors.append(f"{provider_name}: {exc}")
            continue

        configured_provider_count += 1
        successful_provider_count += 1
        offers.extend(provider_offers)

    deduped_offers = dedupe_offers(offers)
    if deduped_offers:
        return deduped_offers

    if configured_provider_count == 0:
        raise ConfigError(
            "No hay proveedores de empleo configurados. "
            "Activa SAMPLE_JOBS_ENABLED=true para datos demo, o configura JOOBLE_API_KEY, "
            "CAREERJET_API_KEY, ADZUNA_APP_ID/ADZUNA_APP_KEY o THEIRSTACK_API_KEY."
        )

    if successful_provider_count > 0:
        return []

    if external_errors:
        raise ExternalServiceError(
            "No pude consultar ningun proveedor configurado: "
            + "; ".join(external_errors)
        )

    return []


def resolve_job_providers(
    providers: Sequence[str] | None = None,
    *,
    credentials: Mapping[str, str | None] | None = None,
) -> tuple[str, ...]:
    raw_providers: Sequence[str]
    if providers is None:
        configured = _resolve_setting("JOB_PROVIDERS", credentials=credentials)
        raw_providers = configured.split(",") if configured else DEFAULT_JOB_PROVIDERS
    else:
        raw_providers = providers

    resolved: list[str] = []
    for provider in raw_providers:
        normalized = str(provider or "").strip().lower()
        if not normalized:
            continue
        if normalized == "all":
            resolved.extend(SUPPORTED_JOB_PROVIDERS)
            continue
        if normalized not in SUPPORTED_JOB_PROVIDERS:
            raise ConfigError(
                f"Proveedor de empleo no soportado: {normalized}. "
                f"Usa alguno de: {', '.join(SUPPORTED_JOB_PROVIDERS)}."
            )
        resolved.append(normalized)

    if providers is None and sample_jobs_enabled(credentials=credentials):
        resolved.insert(0, "sample")

    return tuple(dict.fromkeys(resolved))


def sample_jobs_enabled(
    *,
    credentials: Mapping[str, str | None] | None = None,
) -> bool:
    value = _resolve_setting("SAMPLE_JOBS_ENABLED", credentials=credentials, default="true")
    return value.lower() not in {"0", "false", "no", "off"}


def search_sample_jobs(
    criterio: str,
    *,
    credentials: Mapping[str, str | None] | None = None,
    location: str | None = None,
    remote: bool = False,
    results_per_page: int = DEFAULT_RESULTS_PER_PROVIDER,
    file_path: Path = SAMPLE_JOBS_FILE,
) -> list[dict[str, Any]]:
    query = str(criterio or "").lower().strip()
    if not query:
        return []
    if not sample_jobs_enabled(credentials=credentials):
        raise ConfigError("El proveedor JSON demo esta desactivado.")

    sample_offers = load_sample_jobs(file_path=file_path)
    query_terms = _tokenize_search(query)
    resolved_location = str(location or "").lower().strip()
    matched_offers: list[dict[str, Any]] = []

    for offer in sample_offers:
        searchable_text = " ".join(
            [
                clean_text(offer.get("title"), ""),
                clean_text(_nested_get(offer, "company", "display_name"), ""),
                clean_text(_nested_get(offer, "location", "display_name"), ""),
                clean_text(offer.get("description"), ""),
                clean_text(offer.get("_category"), ""),
                clean_text(offer.get("_role"), ""),
            ]
        ).lower()
        offer_location = clean_text(_nested_get(offer, "location", "display_name"), "").lower()

        if resolved_location and resolved_location not in offer_location:
            continue
        if remote and "remoto" not in searchable_text and "remote" not in searchable_text:
            continue
        if query_terms and not all(term in searchable_text for term in query_terms):
            if not any(term in searchable_text for term in query_terms):
                continue

        matched_offers.append(offer)

    return matched_offers[:results_per_page]


def load_sample_jobs(
    *,
    file_path: Path = SAMPLE_JOBS_FILE,
) -> list[dict[str, Any]]:
    if not file_path.exists():
        raise ConfigError(f"No existe el archivo de ofertas demo: {file_path}")

    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ExternalServiceError("ofertas.json tiene un JSON invalido.") from exc

    if not isinstance(payload, list):
        raise ExternalServiceError("ofertas.json debe contener una lista de ofertas.")

    return [
        _normalize_sample_offer(offer)
        for offer in payload
        if isinstance(offer, Mapping)
    ]


def get_adzuna_credentials(
    app_id: str | None = None,
    app_key: str | None = None,
    credentials: Mapping[str, str | None] | None = None,
) -> tuple[str, str]:
    resolved_app_id = (
        app_id or _resolve_setting("ADZUNA_APP_ID", credentials=credentials)
    ).strip()
    resolved_app_key = (
        app_key or _resolve_setting("ADZUNA_APP_KEY", credentials=credentials)
    ).strip()

    if not resolved_app_id or not resolved_app_key:
        raise ConfigError(
            "Faltan ADZUNA_APP_ID y ADZUNA_APP_KEY. Configuralas en el entorno o en .env."
        )

    return resolved_app_id, resolved_app_key


def search_adzuna(
    criterio: str,
    *,
    app_id: str | None = None,
    app_key: str | None = None,
    credentials: Mapping[str, str | None] | None = None,
    location: str | None = None,
    remote: bool = False,
    posted_at_max_age_days: int | None = None,
    results_per_page: int = DEFAULT_RESULTS_PER_PROVIDER,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> list[dict[str, Any]]:
    query = str(criterio or "").lower().strip()
    if not query:
        return []

    resolved_app_id, resolved_app_key = get_adzuna_credentials(
        app_id, app_key, credentials=credentials
    )
    params = {
        "app_id": resolved_app_id,
        "app_key": resolved_app_key,
        "what": _with_remote_keyword(query, remote),
        "results_per_page": results_per_page,
        "content-type": "application/json",
    }
    resolved_location = str(location or "").strip()
    if resolved_location:
        params["where"] = resolved_location
    if posted_at_max_age_days:
        params["max_days_old"] = posted_at_max_age_days

    try:
        response = requests.get(ADZUNA_URL, params=params, timeout=timeout)
        response.raise_for_status()
    except requests.Timeout as exc:
        raise ExternalServiceError("La búsqueda en Adzuna tardó demasiado.") from exc
    except requests.RequestException as exc:
        raise ExternalServiceError(f"No pude conectar con Adzuna: {exc}") from exc

    try:
        payload = response.json()
    except ValueError as exc:
        raise ExternalServiceError("Adzuna respondió con datos inválidos.") from exc

    results = payload.get("results", [])
    if not isinstance(results, list):
        raise ExternalServiceError("Adzuna respondió con un formato inesperado.")

    return [_normalize_adzuna_offer(offer) for offer in results if isinstance(offer, Mapping)]


def search_jooble(
    criterio: str,
    *,
    api_key: str | None = None,
    credentials: Mapping[str, str | None] | None = None,
    location: str | None = None,
    remote: bool = False,
    results_per_page: int = DEFAULT_RESULTS_PER_PROVIDER,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> list[dict[str, Any]]:
    query = str(criterio or "").lower().strip()
    if not query:
        return []

    resolved_api_key = (
        api_key or _resolve_setting("JOOBLE_API_KEY", credentials=credentials)
    ).strip()
    if not resolved_api_key:
        raise ConfigError("Falta JOOBLE_API_KEY. Configurala en el entorno o en .env.")

    resolved_location = str(location or "").strip() or _resolve_setting(
        "JOOBLE_LOCATION", credentials=credentials, default="Spain"
    )
    payload: dict[str, Any] = {
        "keywords": _with_remote_keyword(query, remote),
        "location": resolved_location,
        "page": "1",
        "ResultOnPage": str(results_per_page),
    }

    try:
        response = requests.post(
            JOOBLE_URL_TEMPLATE.format(api_key=resolved_api_key),
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
    except requests.Timeout as exc:
        raise ExternalServiceError("La busqueda en Jooble tardo demasiado.") from exc
    except requests.RequestException as exc:
        raise ExternalServiceError(f"No pude conectar con Jooble: {exc}") from exc

    try:
        payload = response.json()
    except ValueError as exc:
        raise ExternalServiceError("Jooble respondio con datos invalidos.") from exc

    results = payload.get("jobs", [])
    if not isinstance(results, list):
        raise ExternalServiceError("Jooble respondio con un formato inesperado.")

    return [_normalize_jooble_offer(offer) for offer in results if isinstance(offer, Mapping)]


def search_careerjet(
    criterio: str,
    *,
    api_key: str | None = None,
    credentials: Mapping[str, str | None] | None = None,
    location: str | None = None,
    remote: bool = False,
    results_per_page: int = DEFAULT_RESULTS_PER_PROVIDER,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> list[dict[str, Any]]:
    query = str(criterio or "").lower().strip()
    if not query:
        return []

    resolved_api_key = (
        api_key or _resolve_setting("CAREERJET_API_KEY", credentials=credentials)
    ).strip()
    if not resolved_api_key:
        raise ConfigError(
            "Falta CAREERJET_API_KEY. Configurala en el entorno o en .env."
        )

    params: dict[str, Any] = {
        "locale_code": _resolve_setting(
            "CAREERJET_LOCALE", credentials=credentials, default="es_ES"
        ),
        "keywords": _with_remote_keyword(query, remote),
        "page": 1,
        "page_size": results_per_page,
        "sort": "date",
        "user_ip": _resolve_setting(
            "CAREERJET_USER_IP", credentials=credentials, default="127.0.0.1"
        ),
        "user_agent": _resolve_setting(
            "CAREERJET_USER_AGENT",
            credentials=credentials,
            default="TiburoncinJobBot/1.0",
        ),
    }
    resolved_location = str(location or "").strip() or _resolve_setting(
        "CAREERJET_LOCATION", credentials=credentials
    )
    if resolved_location:
        params["location"] = resolved_location

    try:
        response = requests.get(
            CAREERJET_URL,
            params=params,
            auth=(resolved_api_key, ""),
            timeout=timeout,
        )
        response.raise_for_status()
    except requests.Timeout as exc:
        raise ExternalServiceError("La busqueda en Careerjet tardo demasiado.") from exc
    except requests.RequestException as exc:
        raise ExternalServiceError(f"No pude conectar con Careerjet: {exc}") from exc

    try:
        payload = response.json()
    except ValueError as exc:
        raise ExternalServiceError("Careerjet respondio con datos invalidos.") from exc

    results = payload.get("jobs", [])
    if not isinstance(results, list):
        if payload.get("type") == "LOCATIONS":
            return []
        raise ExternalServiceError("Careerjet respondio con un formato inesperado.")

    return [
        _normalize_careerjet_offer(offer)
        for offer in results
        if isinstance(offer, Mapping)
    ]


def search_theirstack(
    criterio: str,
    *,
    api_key: str | None = None,
    credentials: Mapping[str, str | None] | None = None,
    location: str | None = None,
    remote: bool = False,
    posted_at_max_age_days: int | None = None,
    results_per_page: int = DEFAULT_RESULTS_PER_PROVIDER,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> list[dict[str, Any]]:
    query = str(criterio or "").lower().strip()
    if not query:
        return []

    resolved_api_key = (
        api_key or _resolve_setting("THEIRSTACK_API_KEY", credentials=credentials)
    ).strip()
    if not resolved_api_key:
        raise ConfigError(
            "Falta THEIRSTACK_API_KEY. Configurala en el entorno o en .env."
        )

    country_code = _resolve_setting(
        "THEIRSTACK_COUNTRY_CODE", credentials=credentials, default="ES"
    )
    max_age_days = posted_at_max_age_days or _resolve_positive_int(
        _resolve_setting(
            "THEIRSTACK_POSTED_AT_MAX_AGE_DAYS",
            credentials=credentials,
            default="30",
        ),
        default=30,
    )
    payload = {
        "job_title_or": [query],
        "job_country_code_or": [country_code],
        "posted_at_max_age_days": max_age_days,
        "limit": results_per_page,
        "page": 0,
    }
    resolved_location = str(location or "").strip()
    if resolved_location:
        payload["job_location_pattern_or"] = [resolved_location]
    if remote:
        payload["workplace_types_or"] = ["remote"]
    headers = {
        "Authorization": f"Bearer {resolved_api_key}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(
            THEIRSTACK_URL,
            json=payload,
            headers=headers,
            timeout=timeout,
        )
        response.raise_for_status()
    except requests.Timeout as exc:
        raise ExternalServiceError(
            "La busqueda en TheirStack tardo demasiado."
        ) from exc
    except requests.RequestException as exc:
        raise ExternalServiceError(f"No pude conectar con TheirStack: {exc}") from exc

    try:
        payload = response.json()
    except ValueError as exc:
        raise ExternalServiceError("TheirStack respondio con datos invalidos.") from exc

    results = payload.get("data", [])
    if not isinstance(results, list):
        raise ExternalServiceError("TheirStack respondio con un formato inesperado.")

    return [
        _normalize_theirstack_offer(offer)
        for offer in results
        if isinstance(offer, Mapping)
    ]


def format_offers_markdown(ofertas: Sequence[Mapping[str, Any]]) -> str:
    lines = []
    for oferta in ofertas:
        titulo = clean_text(oferta.get("title"), "Puesto N/A")
        empresa = clean_text(
            _nested_get(oferta, "company", "display_name"), "Empresa N/A"
        )
        lugar = clean_text(_nested_get(oferta, "location", "display_name"), "N/A")
        link = clean_text(oferta.get("redirect_url"), "#")
        provider = clean_text(oferta.get("_provider"), "")
        salary = clean_text(oferta.get("_salary"), "")
        provider_prefix = f"[{provider}] " if provider else ""
        salary_suffix = f" - {salary}" if salary else ""
        lines.append(
            f"- {provider_prefix}{titulo} en {empresa} ({lugar})"
            f"{salary_suffix}. [Enlace]({link})"
        )

    return "\n".join(lines)


def dedupe_offers(ofertas: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    deduped = []

    for oferta in ofertas:
        if not isinstance(oferta, Mapping):
            continue

        link = clean_text(oferta.get("redirect_url"), "")
        title = clean_text(oferta.get("title"), "")
        company = clean_text(_nested_get(oferta, "company", "display_name"), "")
        location = clean_text(_nested_get(oferta, "location", "display_name"), "")
        key = (link or f"{title}|{company}|{location}").lower()
        if not key or key in seen:
            continue

        seen.add(key)
        deduped.append(dict(oferta))

    return deduped


def save_favorite(
    detalle: str,
    enlace_url: str | None = None,
    *,
    file_path: Path = FAVORITES_FILE,
) -> Path:
    cleaned_detail = _single_line(detalle) or "Sin detalle"
    cleaned_link = _single_line(enlace_url or "")
    fecha_hoy = datetime.now().strftime("%d/%m/%Y")

    if cleaned_link:
        line = (
            f"POSTULACIÓN PENDIENTE ({fecha_hoy}): "
            f"{cleaned_detail} - Link: {cleaned_link}\n"
        )
    else:
        line = f"POSTULACIÓN PENDIENTE ({fecha_hoy}): {cleaned_detail}\n"

    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("a", encoding="utf-8") as favorite_file:
        favorite_file.write(line)

    return file_path


def normalize_tool_args(args: Mapping[str, Any]) -> dict[str, Any]:
    normalized = {}
    for key, value in args.items():
        if isinstance(value, dict) and "value" in value:
            normalized[key] = value["value"]
        else:
            normalized[key] = value
    return normalized


def clean_text(value: Any, default: str) -> str:
    if value is None:
        return default

    text = html.unescape(_TAG_RE.sub("", str(value))).strip()
    return text or default


def _single_line(value: str) -> str:
    text = str(value or "").replace("\r", " ").replace("\n", " ").strip()
    return " ".join(text.split())


def _nested_get(mapping: Mapping[str, Any], *keys: str) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _normalize_adzuna_offer(offer: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(offer)
    normalized.setdefault("_provider", "Adzuna")
    return normalized


def _normalize_jooble_offer(offer: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "title": offer.get("title"),
        "company": {"display_name": offer.get("company")},
        "location": {"display_name": offer.get("location")},
        "redirect_url": offer.get("link"),
        "description": offer.get("snippet"),
        "_provider": "Jooble",
        "_source": offer.get("source"),
        "_source_id": offer.get("id"),
        "_salary": offer.get("salary"),
        "_updated": offer.get("updated"),
    }


def _normalize_careerjet_offer(offer: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "title": offer.get("title"),
        "company": {"display_name": offer.get("company")},
        "location": {"display_name": offer.get("locations")},
        "redirect_url": offer.get("url"),
        "description": offer.get("description"),
        "_provider": "Careerjet",
        "_salary": offer.get("salary"),
        "_updated": offer.get("date"),
    }


def _normalize_theirstack_offer(offer: Mapping[str, Any]) -> dict[str, Any]:
    company = offer.get("company") or _nested_get(offer, "company_object", "name")
    cities = offer.get("cities") if isinstance(offer.get("cities"), list) else []
    location = (
        offer.get("long_location")
        or offer.get("location")
        or ", ".join(str(city) for city in cities[:2])
    )
    link = offer.get("final_url") or offer.get("url") or offer.get("source_url")
    return {
        "title": offer.get("job_title"),
        "company": {"display_name": company},
        "location": {"display_name": location},
        "redirect_url": link,
        "description": offer.get("description"),
        "_provider": "TheirStack",
        "_source_id": offer.get("id"),
        "_salary": offer.get("salary_string"),
        "_updated": offer.get("date_posted") or offer.get("discovered_at"),
    }


def _normalize_sample_offer(offer: Mapping[str, Any]) -> dict[str, Any]:
    title = offer.get("title") or offer.get("puesto")
    company = offer.get("company")
    if isinstance(company, Mapping):
        company_name = company.get("display_name")
    else:
        company_name = company

    location = offer.get("location")
    if isinstance(location, Mapping):
        location_name = location.get("display_name")
    else:
        location_name = location or offer.get("ciudad")

    return {
        "title": title,
        "company": {"display_name": company_name or offer.get("empresa") or "Demo Corp"},
        "location": {"display_name": location_name},
        "redirect_url": offer.get("redirect_url")
        or offer.get("url")
        or f"https://example.test/jobs/{offer.get('id', 'demo')}",
        "description": offer.get("description") or offer.get("descripcion"),
        "_provider": "JSON demo",
        "_source_id": offer.get("id"),
        "_salary": offer.get("_salary") or offer.get("salary") or offer.get("sueldo"),
        "_updated": offer.get("_updated") or offer.get("updated") or "Demo",
        "_category": offer.get("_category") or offer.get("categoria"),
        "_role": offer.get("_role") or offer.get("role") or offer.get("puesto"),
    }


def _resolve_setting(
    name: str,
    *,
    credentials: Mapping[str, str | None] | None = None,
    default: str = "",
) -> str:
    value = credentials.get(name) if credentials else None
    if value is None:
        value = os.getenv(name, default)
    return str(value or "").strip()


def _resolve_positive_int(value: str, *, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _tokenize_search(query: str) -> list[str]:
    tokens = re.findall(r"[\wáéíóúüñ]+", query.lower(), flags=re.IGNORECASE)
    stop_words = {
        "a",
        "al",
        "and",
        "de",
        "del",
        "el",
        "en",
        "for",
        "la",
        "las",
        "los",
        "of",
        "para",
        "the",
        "y",
    }
    return [token for token in tokens if token not in stop_words]


def _with_remote_keyword(query: str, remote: bool) -> str:
    if not remote:
        return query

    normalized_query = query.lower()
    remote_words = ("remoto", "teletrabajo", "remote")
    if any(word in normalized_query for word in remote_words):
        return query

    return f"{query} remoto"
