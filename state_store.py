from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence, TypeVar

from empleo_utils import FAVORITES_FILE, clean_text


FAVORITES_JSON_FILE = FAVORITES_FILE.with_suffix(".json")
SAVED_SEARCHES_FILE = FAVORITES_FILE.with_name("busquedas_guardadas.json")
STATUS_OPTIONS = ("pending", "applied", "interview", "discarded")

_Record = TypeVar("_Record", bound=Mapping[str, Any])


class StateFileError(RuntimeError):
    """Raised when local app state cannot be read safely."""


def stable_id(*parts: str) -> str:
    raw = "|".join(str(part or "").strip().lower() for part in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def parse_saved_at(value: Any) -> datetime | None:
    text = clean_text(value, "")
    if not text:
        return None

    candidates = [text]
    if text.endswith("Z"):
        candidates.append(f"{text[:-1]}+00:00")

    for candidate in candidates:
        try:
            parsed = datetime.fromisoformat(candidate)
        except ValueError:
            continue
        if parsed.tzinfo is not None:
            parsed = parsed.astimezone().replace(tzinfo=None)
        return parsed

    for date_format in ("%d/%m/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, date_format)
        except ValueError:
            continue

    return None


def format_saved_at(value: Any) -> str:
    parsed = parse_saved_at(value)
    if parsed is None:
        return ""
    if parsed.hour == 0 and parsed.minute == 0 and parsed.second == 0:
        return parsed.strftime("%d/%m/%Y")
    return parsed.strftime("%d/%m/%Y %H:%M")


def legacy_saved_at(raw_favorite: str) -> str:
    patterns = (
        r"POSTULACI[ÓO]N PENDIENTE\s*\((\d{1,2}/\d{1,2}/\d{4})\)",
        r"GUARDADO EL\s+(\d{1,2}/\d{1,2}/\d{4})",
    )
    for pattern in patterns:
        match = re.search(pattern, raw_favorite, flags=re.IGNORECASE)
        if not match:
            continue
        parsed = parse_saved_at(match.group(1))
        if parsed is not None:
            return parsed.isoformat(timespec="seconds")
    return ""


def saved_at_sort_value(record: Mapping[str, Any]) -> datetime:
    return parse_saved_at(record.get("saved_at")) or datetime.min


def newest_first(records: Sequence[_Record]) -> list[_Record]:
    return [
        record
        for _index, record in sorted(
            enumerate(records),
            key=lambda item: (saved_at_sort_value(item[1]), item[0]),
            reverse=True,
        )
    ]


def read_favorites(file_path: Path = FAVORITES_FILE) -> list[str]:
    if not file_path.exists():
        return []
    return [
        line.strip()
        for line in file_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_json_list(file_path: Path, label: str) -> list[Any]:
    if not file_path.exists():
        return []

    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise StateFileError(
            f"{label} contiene JSON invalido: {file_path}. "
            "Corrige o renombra el archivo antes de guardar cambios."
        ) from exc

    if not isinstance(payload, list):
        raise StateFileError(
            f"{label} debe contener una lista JSON: {file_path}. "
            "Corrige o renombra el archivo antes de guardar cambios."
        )

    return payload


def atomic_write_text(file_path: Path, content: str) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=file_path.parent,
            prefix=f".{file_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_file.write(content)
            temp_file.flush()
            os.fsync(temp_file.fileno())
            temp_path = Path(temp_file.name)
        temp_path.replace(file_path)
    except Exception:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise


def normalize_favorite_record(
    record: Mapping[str, Any],
    *,
    default_title: str = "Favoritos",
    saved_at_default: str | None = None,
    status_options: Sequence[str] = STATUS_OPTIONS,
) -> dict[str, str]:
    if saved_at_default is None:
        saved_at_default = now_iso()

    title = clean_text(record.get("title") or record.get("text"), default_title)
    company = clean_text(record.get("company"), "")
    location = clean_text(record.get("location"), "")
    link = clean_text(record.get("link") or record.get("url"), "")
    record_id = clean_text(record.get("id"), "") or stable_id(title, company, location, link)
    status = clean_text(record.get("status"), "pending")
    if status not in status_options:
        status = "pending"

    return {
        "id": record_id,
        "title": title,
        "company": company,
        "location": location,
        "provider": clean_text(record.get("provider"), ""),
        "salary": clean_text(record.get("salary"), ""),
        "description": clean_text(record.get("description"), ""),
        "link": link,
        "status": status,
        "notes": clean_text(record.get("notes"), ""),
        "saved_at": clean_text(record.get("saved_at"), saved_at_default),
    }


def load_json_favorites(
    file_path: Path = FAVORITES_JSON_FILE,
    *,
    default_title: str = "Favoritos",
) -> list[dict[str, str]]:
    payload = load_json_list(file_path, "favoritos.json")

    records = []
    for item in payload:
        if isinstance(item, Mapping):
            records.append(
                normalize_favorite_record(
                    item,
                    default_title=default_title,
                    saved_at_default="",
                )
            )
    return records


def write_json_favorites(
    favorites: list[Mapping[str, Any]],
    file_path: Path = FAVORITES_JSON_FILE,
    *,
    default_title: str = "Favoritos",
) -> None:
    normalized = [
        normalize_favorite_record(favorite, default_title=default_title)
        for favorite in favorites
    ]
    atomic_write_text(
        file_path,
        json.dumps(normalized, ensure_ascii=False, indent=2),
    )


def parse_favorite(
    raw_favorite: str,
    *,
    default_title: str = "Favoritos",
) -> dict[str, str]:
    text = str(raw_favorite or "").strip()
    link = ""
    saved_at = legacy_saved_at(text)

    markdown_link = re.search(
        r"\[(?:Enlace|Link|Abrir oferta|Open job)\]\((https?://[^)]+)\)",
        text,
        flags=re.IGNORECASE,
    )
    if markdown_link:
        link = markdown_link.group(1).strip()
        text = re.sub(
            r"\s*\.?\s*\[(?:Enlace|Link|Abrir oferta|Open job)\]\(https?://[^)]+\)",
            "",
            text,
            flags=re.IGNORECASE,
        ).strip()

    link_suffix = re.search(r"\s+-\s+Link:\s*(https?://\S+)\s*$", text, flags=re.IGNORECASE)
    if link_suffix:
        link = link_suffix.group(1).strip()
        text = text[: link_suffix.start()].strip()

    text = re.sub(
        r"^(?:POSTULACIÓN PENDIENTE(?:\s+\([^)]+\))?|GUARDADO EL\s+[^:]+|Guardado):\s*",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()

    return normalize_favorite_record(
        {
            "id": stable_id(text or default_title, link),
            "title": text or default_title,
            "link": link,
            "status": "pending",
            "saved_at": saved_at,
        },
        default_title=default_title,
        saved_at_default="",
    )


def legacy_favorite_records(
    file_path: Path = FAVORITES_FILE,
    *,
    default_title: str = "Favoritos",
) -> list[dict[str, str]]:
    return [
        parse_favorite(favorite, default_title=default_title)
        for favorite in read_favorites(file_path=file_path)
    ]


def read_favorite_records(
    *,
    json_file_path: Path = FAVORITES_JSON_FILE,
    legacy_file_path: Path = FAVORITES_FILE,
    default_title: str = "Favoritos",
) -> list[dict[str, str]]:
    json_records = load_json_favorites(
        file_path=json_file_path,
        default_title=default_title,
    )
    seen = {record["id"] for record in json_records}
    records = list(json_records)

    for record in legacy_favorite_records(
        file_path=legacy_file_path,
        default_title=default_title,
    ):
        if record["id"] in seen:
            continue
        seen.add(record["id"])
        records.append(record)

    return newest_first(records)


def merge_favorite_record(
    current: Mapping[str, str],
    normalized: Mapping[str, str],
    *,
    preserve_existing_metadata: bool,
    default_title: str = "Favoritos",
) -> dict[str, str]:
    merged = {**current, **normalized}
    if preserve_existing_metadata:
        for key in ("status", "notes", "saved_at"):
            if current.get(key):
                merged[key] = current[key]
    return normalize_favorite_record(merged, default_title=default_title)


def upsert_favorite_record(
    record: Mapping[str, Any],
    *,
    preserve_existing_metadata: bool = True,
    file_path: Path = FAVORITES_JSON_FILE,
    default_title: str = "Favoritos",
) -> dict[str, str]:
    normalized = normalize_favorite_record(record, default_title=default_title)
    records = load_json_favorites(file_path=file_path, default_title=default_title)
    updated = False

    for index, current in enumerate(records):
        if current["id"] == normalized["id"]:
            normalized = merge_favorite_record(
                current,
                normalized,
                preserve_existing_metadata=preserve_existing_metadata,
                default_title=default_title,
            )
            records[index] = normalized
            updated = True
            break

    if not updated:
        records.append(normalized)

    write_json_favorites(records, file_path=file_path, default_title=default_title)
    return normalized


def update_favorite_record(
    record: Mapping[str, Any],
    *,
    file_path: Path = FAVORITES_JSON_FILE,
    default_title: str = "Favoritos",
    **updates: str,
) -> dict[str, str]:
    normalized = normalize_favorite_record(
        {**record, **updates},
        default_title=default_title,
    )
    return upsert_favorite_record(
        normalized,
        preserve_existing_metadata=False,
        file_path=file_path,
        default_title=default_title,
    )


def delete_favorite_record(
    record_id: str,
    *,
    file_path: Path = FAVORITES_JSON_FILE,
    legacy_file_path: Path = FAVORITES_FILE,
    default_title: str = "Favoritos",
) -> bool:
    records = load_json_favorites(file_path=file_path, default_title=default_title)
    filtered = [record for record in records if record["id"] != record_id]
    json_changed = len(filtered) != len(records)
    if json_changed:
        write_json_favorites(filtered, file_path=file_path, default_title=default_title)

    legacy_lines = read_favorites(file_path=legacy_file_path)
    filtered_legacy = [
        line
        for line in legacy_lines
        if parse_favorite(line, default_title=default_title)["id"] != record_id
    ]
    legacy_changed = len(filtered_legacy) != len(legacy_lines)
    if legacy_changed:
        content = "\n".join(filtered_legacy)
        atomic_write_text(legacy_file_path, f"{content}\n" if content else "")

    return json_changed or legacy_changed


def favorite_search_text(favorite: Mapping[str, Any]) -> str:
    return " ".join(
        clean_text(favorite.get(key), "")
        for key in (
            "title",
            "company",
            "location",
            "provider",
            "salary",
            "description",
            "notes",
            "link",
            "saved_at",
            "status",
        )
    ).lower()


def filter_favorite_records(
    favorites: Sequence[Mapping[str, str]],
    *,
    text: str = "",
    status: str = "",
    providers: Sequence[str] = (),
    locations: Sequence[str] = (),
) -> list[dict[str, str]]:
    terms = [term for term in str(text or "").lower().split() if term]
    provider_set = set(providers)
    location_set = set(locations)
    filtered: list[dict[str, str]] = []

    for favorite in favorites:
        if status and favorite.get("status", "pending") != status:
            continue
        if provider_set and favorite.get("provider", "") not in provider_set:
            continue
        if location_set and favorite.get("location", "") not in location_set:
            continue
        searchable = favorite_search_text(favorite)
        if terms and not all(term in searchable for term in terms):
            continue
        filtered.append(dict(favorite))

    return filtered


def favorite_ids(favorites: Sequence[Mapping[str, str]]) -> set[str]:
    return {favorite["id"] for favorite in favorites if favorite.get("id")}


def load_saved_searches(file_path: Path = SAVED_SEARCHES_FILE) -> list[dict[str, Any]]:
    payload = load_json_list(file_path, "busquedas_guardadas.json")
    return [item for item in payload if isinstance(item, dict)]


def write_saved_searches(
    searches: list[Mapping[str, Any]],
    file_path: Path = SAVED_SEARCHES_FILE,
) -> None:
    atomic_write_text(
        file_path,
        json.dumps(list(searches), ensure_ascii=False, indent=2),
    )


def normalize_saved_search(search: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": stable_id(
            str(search.get("query", "")),
            str(search.get("location", "")),
            str(search.get("remote", "")),
            ",".join(search.get("providers", [])),
        ),
        "query": str(search.get("query", "")).strip(),
        "location": str(search.get("location", "")).strip(),
        "remote": bool(search.get("remote", False)),
        "max_age_days": int(search.get("max_age_days", 30)),
        "results_per_provider": int(search.get("results_per_provider", 5)),
        "providers": list(search.get("providers", [])),
        "saved_at": now_iso(),
    }


def save_search(
    search: Mapping[str, Any],
    *,
    file_path: Path = SAVED_SEARCHES_FILE,
) -> dict[str, Any]:
    normalized = normalize_saved_search(search)
    searches = load_saved_searches(file_path=file_path)
    searches = [item for item in searches if item.get("id") != normalized["id"]]
    searches.append(normalized)
    write_saved_searches(searches, file_path=file_path)
    return normalized
