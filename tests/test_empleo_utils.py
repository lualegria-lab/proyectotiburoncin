from pathlib import Path
from unittest import TestCase
from unittest.mock import Mock, patch

import requests

from empleo_utils import (
    ADZUNA_URL,
    CAREERJET_URL,
    ConfigError,
    ExternalServiceError,
    JOOBLE_URL_TEMPLATE,
    THEIRSTACK_URL,
    format_offers_markdown,
    load_sample_jobs,
    save_favorite,
    search_adzuna,
    search_careerjet,
    search_jobs,
    search_jooble,
    search_sample_jobs,
    search_theirstack,
)


class EmpleoUtilsTest(TestCase):
    def test_format_offers_cleans_html_and_handles_missing_fields(self):
        result = format_offers_markdown(
            [
                {
                    "title": "<b>Python</b> Developer",
                    "company": {"display_name": "ACME"},
                    "location": {"display_name": "Madrid"},
                    "redirect_url": "https://example.com/job",
                    "_provider": "Jooble",
                    "_salary": "30000 EUR",
                },
                {},
            ]
        )

        self.assertIn("[Jooble]", result)
        self.assertIn("Python Developer en ACME (Madrid)", result)
        self.assertIn("30000 EUR", result)
        self.assertIn("Puesto N/A en Empresa N/A (N/A)", result)

    def test_save_favorite_writes_utf8_single_line(self):
        target = Path("test_favoritos_tmp.txt")
        try:
            save_favorite("Recepcionista\nMadrid", "https://example.com", file_path=target)
            content = target.read_text(encoding="utf-8")
        finally:
            target.unlink(missing_ok=True)

        self.assertIn("POSTULACIÓN PENDIENTE", content)
        self.assertIn("Recepcionista Madrid - Link: https://example.com", content)

    def test_load_sample_jobs_supports_legacy_json_shape(self):
        target = Path("test_sample_jobs_tmp.json")
        target.write_text(
            '[{"id": 1, "puesto": "Recepcionista", "ciudad": "Madrid", "sueldo": "22000"}]',
            encoding="utf-8",
        )
        try:
            result = load_sample_jobs(file_path=target)
        finally:
            target.unlink(missing_ok=True)

        self.assertEqual(result[0]["title"], "Recepcionista")
        self.assertEqual(result[0]["location"]["display_name"], "Madrid")
        self.assertEqual(result[0]["_provider"], "JSON demo")

    @patch.dict("os.environ", {"SAMPLE_JOBS_ENABLED": "true"}, clear=True)
    def test_search_sample_jobs_filters_by_query_location_and_remote(self):
        target = Path("test_sample_jobs_tmp.json")
        target.write_text(
            """
            [
              {
                "id": "one",
                "title": "Backend developer",
                "company": "ACME",
                "location": "Madrid",
                "salary": "40000",
                "description": "Python remoto",
                "_category": "informatica",
                "_role": "backend"
              },
              {
                "id": "two",
                "title": "Camarero",
                "company": "Bar",
                "location": "Sevilla",
                "salary": "18000",
                "description": "Sala",
                "_category": "hosteleria",
                "_role": "camarero"
              }
            ]
            """,
            encoding="utf-8",
        )
        try:
            result = search_sample_jobs(
                "backend", location="Madrid", remote=True, file_path=target
            )
        finally:
            target.unlink(missing_ok=True)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["title"], "Backend developer")

    @patch.dict("os.environ", {"SAMPLE_JOBS_ENABLED": "false"}, clear=True)
    def test_search_sample_jobs_can_be_disabled(self):
        with self.assertRaises(ConfigError):
            search_sample_jobs("backend")

    @patch.dict("os.environ", {}, clear=True)
    def test_search_adzuna_requires_credentials(self):
        with self.assertRaises(ConfigError):
            search_adzuna("python")

    @patch.dict(
        "os.environ",
        {"ADZUNA_APP_ID": "app-id", "ADZUNA_APP_KEY": "app-key"},
        clear=True,
    )
    @patch("empleo_utils.requests.get")
    def test_search_adzuna_uses_timeout_and_parses_results(self, get):
        response = Mock()
        response.json.return_value = {"results": [{"title": "Python"}]}
        response.raise_for_status.return_value = None
        get.return_value = response

        result = search_adzuna(
            " Python ",
            location="Madrid",
            remote=True,
            posted_at_max_age_days=7,
        )

        self.assertEqual(result, [{"title": "Python", "_provider": "Adzuna"}])
        get.assert_called_once()
        self.assertEqual(get.call_args.args[0], ADZUNA_URL)
        self.assertEqual(get.call_args.kwargs["timeout"], 10)
        self.assertEqual(get.call_args.kwargs["params"]["what"], "python remoto")
        self.assertEqual(get.call_args.kwargs["params"]["where"], "Madrid")
        self.assertEqual(get.call_args.kwargs["params"]["max_days_old"], 7)
        self.assertEqual(get.call_args.kwargs["params"]["results_per_page"], 5)

    @patch.dict(
        "os.environ",
        {"ADZUNA_APP_ID": "app-id", "ADZUNA_APP_KEY": "app-key"},
        clear=True,
    )
    @patch("empleo_utils.requests.get", side_effect=requests.Timeout)
    def test_search_adzuna_wraps_timeouts(self, _get):
        with self.assertRaises(ExternalServiceError):
            search_adzuna("python")

    @patch("empleo_utils.requests.post")
    def test_search_jooble_posts_payload_and_normalizes_results(self, post):
        response = Mock()
        response.json.return_value = {
            "jobs": [
                {
                    "title": "Python Developer",
                    "company": "ACME",
                    "location": "Madrid",
                    "link": "https://example.com/jooble",
                    "salary": "30000 EUR",
                }
            ]
        }
        response.raise_for_status.return_value = None
        post.return_value = response

        result = search_jooble(
            " Python ", api_key="jooble-key", location="Valencia", remote=True
        )

        self.assertEqual(result[0]["_provider"], "Jooble")
        self.assertEqual(result[0]["company"]["display_name"], "ACME")
        self.assertEqual(result[0]["redirect_url"], "https://example.com/jooble")
        post.assert_called_once()
        self.assertEqual(
            post.call_args.args[0], JOOBLE_URL_TEMPLATE.format(api_key="jooble-key")
        )
        self.assertEqual(post.call_args.kwargs["json"]["keywords"], "python remoto")
        self.assertEqual(post.call_args.kwargs["json"]["location"], "Valencia")

    @patch("empleo_utils.requests.get")
    def test_search_careerjet_uses_basic_auth_and_user_context(self, get):
        response = Mock()
        response.json.return_value = {
            "jobs": [
                {
                    "title": "Data Analyst",
                    "company": "ACME",
                    "locations": "Barcelona",
                    "url": "https://example.com/careerjet",
                }
            ]
        }
        response.raise_for_status.return_value = None
        get.return_value = response

        result = search_careerjet(
            "data",
            api_key="careerjet-key",
            credentials={"CAREERJET_USER_IP": "203.0.113.10"},
            location="Barcelona",
            remote=True,
        )

        self.assertEqual(result[0]["_provider"], "Careerjet")
        self.assertEqual(result[0]["location"]["display_name"], "Barcelona")
        get.assert_called_once()
        self.assertEqual(get.call_args.args[0], CAREERJET_URL)
        self.assertEqual(get.call_args.kwargs["auth"], ("careerjet-key", ""))
        self.assertEqual(get.call_args.kwargs["params"]["keywords"], "data remoto")
        self.assertEqual(get.call_args.kwargs["params"]["location"], "Barcelona")
        self.assertEqual(get.call_args.kwargs["params"]["locale_code"], "es_ES")
        self.assertEqual(get.call_args.kwargs["params"]["user_ip"], "203.0.113.10")

    @patch("empleo_utils.requests.post")
    def test_search_theirstack_posts_payload_and_normalizes_results(self, post):
        response = Mock()
        response.json.return_value = {
            "data": [
                {
                    "job_title": "Backend Engineer",
                    "company": "ACME",
                    "long_location": "Madrid, Spain",
                    "final_url": "https://example.com/theirstack",
                    "salary_string": "50000 EUR",
                }
            ]
        }
        response.raise_for_status.return_value = None
        post.return_value = response

        result = search_theirstack(
            "backend",
            api_key="their-key",
            location="Madrid",
            remote=True,
            posted_at_max_age_days=14,
        )

        self.assertEqual(result[0]["_provider"], "TheirStack")
        self.assertEqual(result[0]["company"]["display_name"], "ACME")
        self.assertEqual(result[0]["redirect_url"], "https://example.com/theirstack")
        post.assert_called_once()
        self.assertEqual(post.call_args.args[0], THEIRSTACK_URL)
        self.assertEqual(
            post.call_args.kwargs["headers"]["Authorization"], "Bearer their-key"
        )
        self.assertEqual(post.call_args.kwargs["json"]["job_title_or"], ["backend"])
        self.assertEqual(post.call_args.kwargs["json"]["job_country_code_or"], ["ES"])
        self.assertEqual(post.call_args.kwargs["json"]["posted_at_max_age_days"], 14)
        self.assertEqual(
            post.call_args.kwargs["json"]["job_location_pattern_or"], ["Madrid"]
        )
        self.assertEqual(
            post.call_args.kwargs["json"]["workplace_types_or"], ["remote"]
        )

    def test_search_jobs_skips_missing_providers_and_dedupes(self):
        offer = {
            "title": "Python",
            "company": {"display_name": "ACME"},
            "location": {"display_name": "Madrid"},
            "redirect_url": "https://example.com/job",
            "_provider": "Careerjet",
        }

        with patch("empleo_utils.search_jooble", side_effect=ConfigError("missing")):
            with patch("empleo_utils.search_careerjet", return_value=[offer, offer]):
                with patch(
                    "empleo_utils.search_adzuna", side_effect=ConfigError("missing")
                ):
                    result = search_jobs(
                        "python",
                        providers=["jooble", "careerjet", "adzuna"],
                        location="Madrid",
                        remote=True,
                        posted_at_max_age_days=7,
                    )

        self.assertEqual(result, [offer])

    def test_search_jobs_returns_empty_when_one_provider_succeeds_without_results(self):
        with patch("empleo_utils.search_jooble", return_value=[]):
            with patch("empleo_utils.search_careerjet", side_effect=ConfigError("missing")):
                result = search_jobs("python", providers=["jooble", "careerjet"])

        self.assertEqual(result, [])

    def test_search_jobs_requires_at_least_one_configured_provider(self):
        with self.assertRaises(ConfigError):
            search_jobs(
                "python",
                providers=["adzuna"],
                credentials={"ADZUNA_APP_ID": "", "ADZUNA_APP_KEY": ""},
            )
