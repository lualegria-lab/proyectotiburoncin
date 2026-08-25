import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from state_store import (
    delete_favorite_record,
    filter_favorite_records,
    parse_favorite,
    read_favorite_records,
)


class StateStoreTest(TestCase):
    def test_delete_favorite_record_removes_json_and_matching_legacy_entry(self):
        raw_to_delete = (
            "POSTULACIÓN PENDIENTE: Backend developer - Link: "
            "https://example.com/backend"
        )
        raw_to_keep = (
            "POSTULACIÓN PENDIENTE: Frontend developer - Link: "
            "https://example.com/frontend"
        )
        record_to_delete = parse_favorite(raw_to_delete)

        with TemporaryDirectory() as temp_dir:
            json_target = Path(temp_dir) / "favoritos.json"
            legacy_target = Path(temp_dir) / "favoritos.txt"
            json_target.write_text(
                json.dumps(
                    [
                        {
                            **record_to_delete,
                            "company": "ACME",
                            "provider": "Jooble",
                        },
                        {
                            "id": "keep-json",
                            "title": "Data analyst",
                            "saved_at": "2026-08-18T12:00:00",
                        },
                    ]
                ),
                encoding="utf-8",
            )
            legacy_target.write_text(f"{raw_to_delete}\n{raw_to_keep}\n", encoding="utf-8")

            deleted = delete_favorite_record(
                record_to_delete["id"],
                file_path=json_target,
                legacy_file_path=legacy_target,
            )
            records = read_favorite_records(
                json_file_path=json_target,
                legacy_file_path=legacy_target,
            )

        self.assertTrue(deleted)
        self.assertNotIn(record_to_delete["id"], {record["id"] for record in records})
        self.assertEqual(
            [record["title"] for record in records],
            ["Data analyst", "Frontend developer"],
        )

    def test_filter_favorite_records_combines_text_status_provider_and_location(self):
        favorites = [
            {
                "id": "one",
                "title": "Python developer",
                "company": "ACME",
                "location": "Madrid",
                "provider": "Adzuna",
                "status": "pending",
                "notes": "remote first",
            },
            {
                "id": "two",
                "title": "Python developer",
                "company": "Beta",
                "location": "Barcelona",
                "provider": "Jooble",
                "status": "pending",
                "notes": "",
            },
            {
                "id": "three",
                "title": "Marketing manager",
                "company": "Gamma",
                "location": "Madrid",
                "provider": "Adzuna",
                "status": "applied",
                "notes": "",
            },
        ]

        result = filter_favorite_records(
            favorites,
            text="python remote",
            status="pending",
            providers=["Adzuna"],
            locations=["Madrid"],
        )

        self.assertEqual([favorite["id"] for favorite in result], ["one"])
