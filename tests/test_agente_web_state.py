import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from langchain_core.messages import AIMessage

from agente_web import (
    _chat_system_message,
    _detect_chat_language,
    _format_offers_markdown,
    _resolve_chat_offer_reference,
    _save_chat_offer_reference,
    _visible_ai_content,
)
from state_store import (
    StateFileError,
    load_json_favorites as _load_json_favorites,
    load_saved_searches as _load_saved_searches,
    read_favorite_records as _read_favorite_records,
    upsert_favorite_record as _upsert_favorite_record,
)


class AgenteWebStateTest(TestCase):
    def test_chat_language_is_detected_from_each_prompt(self):
        self.assertEqual(_detect_chat_language("Can you find remote marketing jobs?"), "en")
        self.assertEqual(_detect_chat_language("¿Puedes buscar empleos remotos?"), "es")
        self.assertEqual(_detect_chat_language("Frontend Madrid", "en"), "en")

    def test_chat_system_message_uses_detected_language(self):
        english_message = _chat_system_message("en").content
        spanish_message = _chat_system_message("es").content

        self.assertIn("Respond only in English", english_message)
        self.assertIn("Respond only in Spanish", spanish_message)

    def test_internal_tool_content_is_hidden_from_chat_history(self):
        raw_tool_json = AIMessage(
            content='{"name": "buscar_empleos", "parameters": {"criterio": "Madrid"}}'
        )
        tool_call_message = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "buscar_empleos",
                    "args": {"criterio": "Madrid"},
                    "id": "call-1",
                    "type": "tool_call",
                }
            ],
        )
        normal_message = AIMessage(content="He encontrado varias ofertas en Madrid.")

        self.assertEqual(_visible_ai_content(raw_tool_json), "")
        self.assertEqual(_visible_ai_content(tool_call_message), "")
        self.assertEqual(
            _visible_ai_content(normal_message),
            "He encontrado varias ofertas en Madrid.",
        )

    def test_chat_offer_reference_resolves_by_number_word_and_text(self):
        offers = [
            {
                "title": "Backend developer",
                "company": {"display_name": "ACME"},
                "location": {"display_name": "Madrid"},
                "redirect_url": "https://example.com/backend",
                "_provider": "Adzuna",
            },
            {
                "title": "Frontend developer",
                "company": {"display_name": "Blue Pixel"},
                "location": {"display_name": "Barcelona"},
                "redirect_url": "https://example.com/frontend",
                "_provider": "JSON demo",
            },
        ]

        with patch("agente_web._t", return_value="N/A"):
            self.assertEqual(
                _resolve_chat_offer_reference("2", offers=offers)["title"],
                "Frontend developer",
            )
            self.assertEqual(
                _resolve_chat_offer_reference("first", offers=offers)["title"],
                "Backend developer",
            )
            self.assertEqual(
                _resolve_chat_offer_reference("Blue Pixel", offers=offers)["title"],
                "Frontend developer",
            )
            self.assertEqual(
                _resolve_chat_offer_reference(
                    "", "https://example.com/backend", offers=offers
                )["title"],
                "Backend developer",
            )

    def test_chat_offer_markdown_is_numbered_for_follow_up_save_requests(self):
        offers = [
            {
                "title": "Frontend developer",
                "company": {"display_name": "Blue Pixel"},
                "location": {"display_name": "Barcelona"},
                "redirect_url": "https://example.com/frontend",
                "_provider": "JSON demo",
            }
        ]

        with patch(
            "agente_web._t",
            side_effect=lambda key: {"at_word": "en", "open_job": "Abrir oferta"}.get(
                key, key
            ),
        ):
            result = _format_offers_markdown(offers)

        self.assertTrue(result.startswith("1. [JSON demo] Frontend developer"))

    def test_chat_offer_markdown_can_use_detected_english_language(self):
        offers = [
            {
                "title": "Frontend developer",
                "company": {"display_name": "Blue Pixel"},
                "location": {"display_name": "Barcelona"},
                "redirect_url": "https://example.com/frontend",
                "_provider": "JSON demo",
            }
        ]

        with patch("agente_web._language_code", return_value="es"):
            result = _format_offers_markdown(offers, language="en")

        self.assertIn("Frontend developer at Blue Pixel", result)
        self.assertIn("[Open job](https://example.com/frontend)", result)

    def test_chat_offer_reference_can_be_saved_to_favorites_json(self):
        offers = [
            {
                "title": "Backend developer",
                "company": {"display_name": "ACME"},
                "location": {"display_name": "Madrid"},
                "redirect_url": "https://example.com/backend",
                "_provider": "Adzuna",
            },
            {
                "title": "Frontend developer",
                "company": {"display_name": "Blue Pixel"},
                "location": {"display_name": "Barcelona"},
                "redirect_url": "https://example.com/frontend",
                "_provider": "JSON demo",
            },
        ]

        with TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "favoritos.json"
            translations = {
                "job_na": "Puesto N/A",
                "company_na": "Empresa N/A",
                "source_na": "Fuente N/A",
                "chat_saved_offer": "Guardado en favoritos",
                "favorites": "Favoritos",
            }
            with patch("agente_web.st.session_state", {"chat_search_results": offers}):
                with patch("agente_web._t", side_effect=lambda key: translations.get(key, key)):
                    result = _save_chat_offer_reference("2", file_path=target)
                    records = _load_json_favorites(file_path=target)

        self.assertEqual(result, "Guardado en favoritos: Frontend developer")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["title"], "Frontend developer")
        self.assertEqual(records[0]["company"], "Blue Pixel")
        self.assertEqual(records[0]["location"], "Barcelona")
        self.assertEqual(records[0]["link"], "https://example.com/frontend")

    def test_favorites_are_read_from_newest_to_oldest(self):
        with TemporaryDirectory() as temp_dir:
            json_target = Path(temp_dir) / "favoritos.json"
            legacy_target = Path(temp_dir) / "favoritos.txt"
            json_target.write_text(
                json.dumps(
                    [
                        {
                            "id": "old-json",
                            "title": "Old JSON favorite",
                            "saved_at": "2026-08-01T09:30:00",
                        },
                        {
                            "id": "new-json",
                            "title": "New JSON favorite",
                            "saved_at": "2026-08-18T12:00:00",
                        },
                    ]
                ),
                encoding="utf-8",
            )
            legacy_target.write_text(
                "\n".join(
                    [
                        "POSTULACIÓN PENDIENTE: Undated legacy favorite",
                        "GUARDADO EL 15/08/2026: Dated legacy favorite",
                    ]
                ),
                encoding="utf-8",
            )

            with patch("agente_web._t", return_value="Favoritos"):
                records = _read_favorite_records(
                    json_file_path=json_target,
                    legacy_file_path=legacy_target,
                )

        self.assertEqual(
            [record["title"] for record in records],
            [
                "New JSON favorite",
                "Dated legacy favorite",
                "Old JSON favorite",
                "Undated legacy favorite",
            ],
        )

    def test_duplicate_favorite_save_preserves_user_metadata(self):
        with TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "favoritos.json"
            target.write_text(
                json.dumps(
                    [
                        {
                            "id": "job-1",
                            "title": "Backend developer",
                            "company": "ACME",
                            "location": "Madrid",
                            "provider": "Adzuna",
                            "salary": "40000 EUR",
                            "description": "Old description",
                            "link": "https://example.com/job",
                            "status": "interview",
                            "notes": "Call scheduled",
                            "saved_at": "2026-08-01T09:30:00",
                        }
                    ]
                ),
                encoding="utf-8",
            )

            with patch("agente_web._t", return_value="Favoritos"):
                _upsert_favorite_record(
                    {
                        "id": "job-1",
                        "title": "Backend developer",
                        "company": "ACME",
                        "location": "Madrid",
                        "provider": "Adzuna",
                        "salary": "45000 EUR",
                        "description": "Updated description",
                        "link": "https://example.com/job",
                        "status": "pending",
                        "notes": "",
                        "saved_at": "2026-08-18T12:00:00",
                    },
                    file_path=target,
                )
                records = _load_json_favorites(file_path=target)

        self.assertEqual(records[0]["status"], "interview")
        self.assertEqual(records[0]["notes"], "Call scheduled")
        self.assertEqual(records[0]["saved_at"], "2026-08-01T09:30:00")
        self.assertEqual(records[0]["salary"], "45000 EUR")
        self.assertEqual(records[0]["description"], "Updated description")

    def test_explicit_favorite_update_can_change_metadata(self):
        with TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "favoritos.json"
            target.write_text(
                json.dumps(
                    [
                        {
                            "id": "job-1",
                            "title": "Backend developer",
                            "status": "pending",
                            "notes": "",
                            "saved_at": "2026-08-01T09:30:00",
                        }
                    ]
                ),
                encoding="utf-8",
            )

            with patch("agente_web._t", return_value="Favoritos"):
                _upsert_favorite_record(
                    {
                        "id": "job-1",
                        "title": "Backend developer",
                        "status": "applied",
                        "notes": "Sent CV",
                        "saved_at": "2026-08-18T12:00:00",
                    },
                    preserve_existing_metadata=False,
                    file_path=target,
                )
                records = _load_json_favorites(file_path=target)

        self.assertEqual(records[0]["status"], "applied")
        self.assertEqual(records[0]["notes"], "Sent CV")

    def test_invalid_favorites_json_blocks_writes(self):
        with TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "favoritos.json"
            target.write_text("{not-json", encoding="utf-8")

            with patch("agente_web._t", return_value="Favoritos"):
                with self.assertRaises(StateFileError):
                    _load_json_favorites(file_path=target)
                with self.assertRaises(StateFileError):
                    _upsert_favorite_record(
                        {"id": "job-1", "title": "Backend developer"},
                        file_path=target,
                    )

            self.assertEqual(target.read_text(encoding="utf-8"), "{not-json")

    def test_invalid_saved_searches_json_is_not_treated_as_empty(self):
        with TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "busquedas_guardadas.json"
            target.write_text('{"not": "a list"}', encoding="utf-8")

            with self.assertRaises(StateFileError):
                _load_saved_searches(file_path=target)
