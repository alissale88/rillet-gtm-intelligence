"""Tests for the hypothetical ICP scoring engine and the dashboard smoke path."""

from __future__ import annotations

import io
import unittest
from pathlib import Path

import pandas as pd

from scoring import (
    DEFAULT_WEIGHTS,
    EXPORT_COLUMNS,
    build_insights,
    dataset_is_synthetic,
    evaluate,
    export_csv_text,
    export_frame,
    model_notes,
    priority_for,
    read_company_csv,
    score_dataframe,
    word_count,
    DataValidationError,
)

ROOT = Path(__file__).resolve().parent
SAMPLE_PATH = ROOT / "sample_accounts.csv"

EXPECTED_SCORES = {
    "Apexloom": 100,
    "Helix Claims Cloud": 100,
    "Meridian Stack": 100,
    "Northstar Ledger": 100,
    "Copperline Ops": 90,
    "Lumengrid Systems": 85,
    "Cascade Revenue": 80,
    "Cinder Pay": 80,
    "Alloy Transit Finance": 75,
    "Driftcurrent": 75,
    "Ironvale Platforms": 75,
    "Ledgerlane": 75,
    "Clearpath Health": 70,
    "Harborline AI": 70,
    "Thornfield Group": 70,
    "Sable & Co. Partners": 60,
    "Bluehaven Markets": 55,
    "Kinetic Model Lab": 55,
    "Nimbus Clinic OS": 55,
    "Quarry Systems": 50,
    "Fieldnote Analytics": 45,
    "Brightfork Software": 35,
    "Oak & Quarter Advisory": 25,
    "Paperkite Software": 20,
    "Solace Benefits Tech": 20,
    "Summit & Birch Advisors": 15,
    "Vectorwell": 15,
    "Kindling Workflows": 10,
    "Pinnacle Forge": 0,
    "Wren & Hollow Consulting": 0,
}

BANNED_PHRASES = [
    "guaranteed",
    "testimonial",
    "roi",
    "customers save",
    "conversion rate",
    "has a problem",
    "struggling",
    "$",
    "%",
]


def base_row(**overrides):
    row = {
        "Company Name": "Example Co",
        "Industry": "B2B SaaS",
        "Employee Count": 100,
        "Funding Stage": "Series B",
        "Business Model": "Subscription SaaS",
        "Complex Billing": "No",
        "Multiple Legal Entities": "No",
        "Finance Team Hiring": "No",
    }
    row.update(overrides)
    return row


def independent_score(row, weights=None):
    """A second copy of the default rules, so the engine is not grading itself."""
    weights = dict(DEFAULT_WEIGHTS if weights is None else {**DEFAULT_WEIGHTS, **weights})
    total = 0
    if str(row["Complex Billing"]).strip().lower() in {"yes", "y", "true", "1"}:
        total += weights["complex_billing"]
    if str(row["Multiple Legal Entities"]).strip().lower() in {"yes", "y", "true", "1"}:
        total += weights["multiple_entities"]
    if str(row["Finance Team Hiring"]).strip().lower() in {"yes", "y", "true", "1"}:
        total += weights["finance_hiring"]
    employees = row["Employee Count"]
    if isinstance(employees, int) and 50 <= employees <= 1000:
        total += weights["employee_band"]
    stage = str(row["Funding Stage"]).strip().lower().replace("-", " ")
    stage = " ".join(stage.split())
    if stage in {"series a", "series b", "series c", "series d"}:
        total += weights["growth_stage"]
    return max(0, min(100, int(total)))


def band(score):
    if score >= 80:
        return "HIGH PRIORITY"
    if score >= 50:
        return "MEDIUM PRIORITY"
    return "LOW PRIORITY"


class PriorityAndCapTests(unittest.TestCase):
    def test_priority_boundaries(self):
        self.assertEqual(priority_for(0), "LOW PRIORITY")
        self.assertEqual(priority_for(49), "LOW PRIORITY")
        self.assertEqual(priority_for(50), "MEDIUM PRIORITY")
        self.assertEqual(priority_for(79), "MEDIUM PRIORITY")
        self.assertEqual(priority_for(80), "HIGH PRIORITY")
        self.assertEqual(priority_for(100), "HIGH PRIORITY")

    def test_scores_never_exceed_100_across_combinations(self):
        for billing in ("Yes", "No"):
            for entities in ("Yes", "No"):
                for hiring in ("Yes", "No"):
                    for employees in (None, 49, 50, 1000, 1001):
                        for stage in ("", "Seed", "Series A", "Series D", "Series E", "Public"):
                            row = base_row(
                                **{
                                    "Complex Billing": billing,
                                    "Multiple Legal Entities": entities,
                                    "Finance Team Hiring": hiring,
                                    "Employee Count": employees if employees is not None else "",
                                    "Funding Stage": stage,
                                }
                            )
                            result = evaluate(row)
                            score = result["columns"]["ICP Score"]
                            self.assertGreaterEqual(score, 0)
                            self.assertLessEqual(score, 100)
                            self.assertEqual(result["columns"]["Priority"], band(score))
                            self.assertEqual(score, independent_score(row))

    def test_custom_weights_change_the_score_and_still_cap(self):
        row = base_row(
            **{
                "Complex Billing": "Yes",
                "Multiple Legal Entities": "Yes",
                "Finance Team Hiring": "Yes",
            }
        )
        lowered = evaluate(row, {"finance_hiring": 0})
        self.assertEqual(lowered["columns"]["ICP Score"], 80)
        huge = evaluate(
            row,
            {
                "complex_billing": 80,
                "multiple_entities": 80,
                "finance_hiring": 80,
                "employee_band": 80,
                "growth_stage": 80,
            },
        )
        self.assertEqual(huge["columns"]["ICP Score"], 100)
        self.assertIn("capped", huge["columns"]["Score Explanation"].lower())

    def test_employee_and_stage_boundaries(self):
        self.assertEqual(evaluate(base_row(**{"Employee Count": 49}))["columns"]["ICP Score"], 10)
        self.assertEqual(evaluate(base_row(**{"Employee Count": 50, "Funding Stage": "Seed"}))["columns"]["ICP Score"], 15)
        self.assertEqual(evaluate(base_row(**{"Employee Count": 1000, "Funding Stage": "Seed"}))["columns"]["ICP Score"], 15)
        self.assertEqual(evaluate(base_row(**{"Employee Count": 1001, "Funding Stage": "Seed"}))["columns"]["ICP Score"], 0)
        self.assertEqual(evaluate(base_row(**{"Employee Count": 10, "Funding Stage": "Series A"}))["columns"]["ICP Score"], 10)
        self.assertEqual(evaluate(base_row(**{"Employee Count": 10, "Funding Stage": "series-d"}))["columns"]["ICP Score"], 10)
        self.assertEqual(evaluate(base_row(**{"Employee Count": 10, "Funding Stage": "Series E"}))["columns"]["ICP Score"], 0)
        self.assertEqual(evaluate(base_row(**{"Employee Count": 10, "Funding Stage": "Public"}))["columns"]["ICP Score"], 0)


class MissingDataTests(unittest.TestCase):
    def test_blank_fields_score_zero_and_do_not_crash(self):
        frame = pd.DataFrame(
            [
                {
                    "Company Name": "Blank Co",
                    "Industry": None,
                    "Employee Count": None,
                    "Funding Stage": None,
                    "Business Model": None,
                    "Complex Billing": None,
                    "Multiple Legal Entities": None,
                    "Finance Team Hiring": None,
                }
            ]
        )
        scored, warnings = score_dataframe(frame)
        self.assertEqual(len(scored), 1)
        self.assertEqual(int(scored.iloc[0]["ICP Score"]), 0)
        self.assertEqual(scored.iloc[0]["Priority"], "LOW PRIORITY")
        explanation = scored.iloc[0]["Score Explanation"].lower()
        self.assertIn("blank", explanation)
        self.assertIn("not treated as a no", explanation)
        self.assertTrue(warnings)

    def test_unreadable_values_are_warnings_not_crashes(self):
        frame = pd.DataFrame(
            [
                base_row(
                    **{
                        "Company Name": "Messy Co",
                        "Employee Count": "many",
                        "Complex Billing": "maybe",
                        "Funding Stage": "Series B",
                    }
                )
            ]
        )
        scored, warnings = score_dataframe(frame)
        self.assertEqual(int(scored.iloc[0]["ICP Score"]), 10)
        self.assertTrue(any("not recognized" in warning for warning in warnings))
        self.assertIn("could not be read", scored.iloc[0]["Score Explanation"].lower())

    def test_blank_company_names_are_skipped(self):
        frame = pd.DataFrame(
            [
                base_row(**{"Company Name": "  "}),
                base_row(**{"Company Name": "Kept Co", "Complex Billing": "Yes"}),
            ]
        )
        scored, warnings = score_dataframe(frame)
        self.assertEqual(list(scored["Company Name"]), ["Kept Co"])
        self.assertTrue(any("skipped" in warning for warning in warnings))

    def test_all_blank_names_raise_a_clear_error(self):
        frame = pd.DataFrame([base_row(**{"Company Name": ""})])
        with self.assertRaises(DataValidationError):
            score_dataframe(frame)

    def test_missing_columns_name_the_gap(self):
        frame = pd.DataFrame([{"Company Name": "Only Name"}])
        with self.assertRaises(DataValidationError) as caught:
            score_dataframe(frame)
        self.assertIn("Industry", str(caught.exception))

    def test_previous_score_column_is_recomputed(self):
        frame = pd.DataFrame([base_row(**{"Complex Billing": "Yes", "ICP Score": 1, "Priority": "LOW PRIORITY"})])
        scored, _warnings = score_dataframe(frame)
        self.assertEqual(int(scored.iloc[0]["ICP Score"]), independent_score(base_row(**{"Complex Billing": "Yes"})))
        self.assertNotEqual(int(scored.iloc[0]["ICP Score"]), 1)


class RankingAndExportTests(unittest.TestCase):
    def test_high_priority_accounts_appear_first(self):
        scored, warnings = score_dataframe(read_company_csv(SAMPLE_PATH))
        self.assertEqual(warnings, [])
        scores = [int(score) for score in scored["ICP Score"]]
        self.assertEqual(scores, sorted(scores, reverse=True))
        seen_lower = False
        for priority, score in zip(scored["Priority"], scores):
            self.assertEqual(priority, band(score))
            if priority != "HIGH PRIORITY":
                seen_lower = True
                self.assertLess(score, 80)
            elif seen_lower:
                self.fail("A high-priority account appeared after a lower-priority account")
        self.assertEqual(scored.iloc[0]["Company Name"], "Apexloom")
        self.assertEqual(scored.iloc[0]["Priority"], "HIGH PRIORITY")

    def test_sample_scores_and_labels(self):
        scored, _warnings = score_dataframe(read_company_csv(SAMPLE_PATH))
        self.assertEqual(len(scored), 30)
        actual = {row["Company Name"]: int(row["ICP Score"]) for _, row in scored.iterrows()}
        self.assertEqual(actual, EXPECTED_SCORES)
        self.assertTrue(dataset_is_synthetic(read_company_csv(SAMPLE_PATH)))
        notes = read_company_csv(SAMPLE_PATH)["Notes"].astype(str)
        self.assertTrue(notes.str.startswith("SYNTHETIC DEMO DATA").all())

    def test_export_contains_the_required_columns_in_order(self):
        scored, _warnings = score_dataframe(read_company_csv(SAMPLE_PATH))
        exported = export_frame(scored)
        self.assertEqual(list(exported.columns), EXPORT_COLUMNS)
        text = export_csv_text(scored)
        self.assertEqual(text.splitlines()[0], ",".join(EXPORT_COLUMNS))
        self.assertIn("Apexloom", text.splitlines()[1])
        self.assertIn("HYPOTHETICAL DRAFT", text)
        self.assertNotIn("Primary Pain Point", text.splitlines()[0])


class RecommendationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scored, _warnings = score_dataframe(read_company_csv(SAMPLE_PATH))

    def _eval(self, name):
        row = self.scored[self.scored["Company Name"] == name].iloc[0]
        return evaluate(row.to_dict())

    def test_two_strongest_signals_drive_the_angle(self):
        apex = self._eval("Apexloom")
        angle = apex["columns"]["Outreach Angle"].lower()
        email = apex["email_body"].lower()
        self.assertIn("revenue recognition", angle)
        self.assertIn("consolidation", angle)
        self.assertNotIn("scaling finance", angle)
        self.assertIn("revenue recognition", email)
        self.assertIn("multi-entity", email)
        self.assertNotIn("scaling finance", email)
        self.assertIn("hypothetical", email)
        self.assertIn("Apexloom", apex["email_body"])

        drift = self._eval("Driftcurrent")
        drift_text = (drift["columns"]["Outreach Angle"] + " " + drift["email_body"]).lower()
        self.assertIn("revenue recognition", drift_text)
        self.assertIn("scaling finance", drift_text)
        self.assertNotIn("consolidation", drift_text)
        self.assertNotIn("multi-entity", drift_text)

        quarry = self._eval("Quarry Systems")
        quarry_text = (quarry["columns"]["Outreach Angle"] + " " + quarry["email_body"]).lower()
        self.assertIn("consolidation", quarry_text)
        self.assertNotIn("revenue recognition", quarry_text)
        self.assertNotIn("scaling finance", quarry_text)

        fieldnote = self._eval("Fieldnote Analytics")
        field_text = (fieldnote["columns"]["Outreach Angle"] + " " + fieldnote["email_body"]).lower()
        self.assertIn("scaling finance", field_text)
        self.assertNotIn("revenue recognition", field_text)
        self.assertNotIn("consolidation", field_text)

    def test_no_signal_account_does_not_invent_a_pain(self):
        pinnacle = self._eval("Pinnacle Forge")
        email = pinnacle["email_body"].lower()
        angle = pinnacle["columns"]["Outreach Angle"].lower()
        self.assertIn("not a recommendation to send yet", email)
        self.assertNotIn("could benefit", email)
        self.assertIn("do not lead", angle)
        self.assertIn("hypothetical", angle)
        self.assertEqual(pinnacle["columns"]["Recommended Buyer"], "Controller")
        self.assertEqual(self._eval("Apexloom")["columns"]["Recommended Buyer"], "CFO")
        self.assertEqual(self._eval("Northstar Ledger")["columns"]["Recommended Buyer"], "VP Finance")

    def test_every_draft_stays_hypothetical_and_near_75_words(self):
        for _, row in self.scored.iterrows():
            result = evaluate(row.to_dict())
            email = result["email_body"]
            draft = result["columns"]["Draft Email"]
            angle = result["columns"]["Outreach Angle"]
            explanation = result["columns"]["Score Explanation"]
            self.assertIn(row["Company Name"], email)
            self.assertIn("hypothetical", email.lower())
            self.assertIn("HYPOTHETICAL", draft)
            self.assertTrue(angle.lower().startswith("hypothetical"))
            count = word_count(email)
            self.assertGreaterEqual(count, 55, email)
            self.assertLessEqual(count, 100, email)
            self.assertIn(result["columns"]["Recommended Buyer"], {"Controller", "VP Finance", "CFO"})
            blob = " ".join([email, angle, explanation, result["why"]]).lower()
            for phrase in BANNED_PHRASES:
                self.assertNotIn(phrase, blob)

    def test_insights_match_the_sample(self):
        insights = build_insights(self.scored)
        self.assertEqual(insights["total"], 30)
        self.assertEqual(insights["high_n"], 8)
        self.assertEqual(insights["medium_n"], 12)
        self.assertEqual(insights["low_n"], 10)
        self.assertEqual(insights["complex_billing_n"], 15)
        self.assertAlmostEqual(insights["avg_score"], 1705 / 30, places=5)
        self.assertEqual(insights["top_industry"], "Fintech")
        self.assertAlmostEqual(insights["top_industry_avg"], 71.25, places=2)
        self.assertEqual(
            insights["most_common_signal_labels"],
            ["Complex billing", "Multiple legal entities"],
        )
        self.assertEqual(insights["most_common_signal_count"], 8)
        self.assertEqual(insights["medium_gap_labels"], ["Multiple legal entities"])
        self.assertEqual(insights["medium_gap_count"], 7)
        self.assertEqual(len(insights["actions"]), 3)
        self.assertIn("8 of 30 accounts are high priority.", insights["summary_lines"][0])
        joined = " ".join(insights["summary_lines"] + insights["actions"]).lower()
        self.assertNotIn("conversion", joined)
        self.assertNotIn("%", joined)
        self.assertNotIn("$", joined)

    def test_model_note_explains_the_default_high_priority_rule(self):
        notes = " ".join(model_notes()).lower()
        self.assertIn("both complex billing and multiple legal entities", notes)
        loosened = " ".join(model_notes({"complex_billing": 60, "multiple_entities": 60})).lower()
        self.assertNotIn("requires both complex billing", loosened)


class CsvFormatTests(unittest.TestCase):
    def test_case_and_aliases_are_accepted(self):
        text = (
            "company name,Sector,Employees,stage,business type,COMPLEX BILLING,"
            "multiple entities,finance hiring\n"
            "Alias Co,Fintech,80,Series A,Usage-based SaaS,YES,no,y\n"
        )
        scored, _warnings = score_dataframe(read_company_csv(io.StringIO(text)))
        self.assertEqual(scored.iloc[0]["Company Name"], "Alias Co")
        self.assertEqual(int(scored.iloc[0]["ICP Score"]), 30 + 20 + 15 + 10)

    def test_semicolon_files_are_read(self):
        text = (
            "Company Name;Industry;Employee Count;Funding Stage;Business Model;"
            "Complex Billing;Multiple Legal Entities;Finance Team Hiring\n"
            "Semi Co;Fintech;80;Series A;Usage-based SaaS;Yes;No;No\n"
        )
        scored, _warnings = score_dataframe(read_company_csv(io.BytesIO(text.encode("utf-8"))))
        self.assertEqual(scored.iloc[0]["Company Name"], "Semi Co")
        self.assertEqual(int(scored.iloc[0]["ICP Score"]), 55)

    def test_duplicate_headers_are_rejected(self):
        text = "Company Name,Company Name,Industry\nA,B,C\n"
        with self.assertRaises(DataValidationError):
            score_dataframe(read_company_csv(io.StringIO(text)))

    def test_empty_and_broken_files_raise_validation_errors(self):
        with self.assertRaises(DataValidationError):
            read_company_csv(io.BytesIO(b""))
        with self.assertRaises(DataValidationError):
            read_company_csv(io.StringIO('Company Name,Industry\n"Acme,Fintech\n'))
        with self.assertRaises(DataValidationError):
            read_company_csv(SAMPLE_PATH.parent / "missing-file.csv")

    def test_header_only_file_is_rejected(self):
        header = ",".join(
            [
                "Company Name",
                "Industry",
                "Employee Count",
                "Funding Stage",
                "Business Model",
                "Complex Billing",
                "Multiple Legal Entities",
                "Finance Team Hiring",
            ]
        )
        with self.assertRaises(DataValidationError):
            score_dataframe(read_company_csv(io.StringIO(header + "\n")))

    def test_insights_still_run_when_nothing_is_high_priority(self):
        frame = pd.DataFrame(
            [
                base_row(**{"Company Name": "Low One", "Employee Count": 10, "Funding Stage": "Seed"}),
                base_row(**{"Company Name": "Low Two", "Complex Billing": "Yes", "Employee Count": 10, "Funding Stage": "Seed"}),
            ]
        )
        scored, _warnings = score_dataframe(frame)
        insights = build_insights(scored)
        self.assertEqual(insights["high_n"], 0)
        self.assertEqual(len(insights["actions"]), 3)
        self.assertIn("No accounts are high priority", insights["summary_lines"][1])


class DashboardSmokeTest(unittest.TestCase):
    def test_dashboard_renders_sample_insights(self):
        from streamlit.testing.v1 import AppTest

        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
        app.run()
        if app.exception:
            self.fail(app.exception)
        text = _visible_text(app)
        self.assertIn("Rillet GTM Intelligence", text)
        self.assertIn("Growth Team Recommendations", text)
        self.assertIn("8 of 30 accounts are high priority.", text)
        self.assertIn("About This Work Sample", text)
        self.assertIn("Hypothetical ICP", text)
        self.assertIn("Apexloom", text)
        self.assertEqual(app.metric[0].value, "30")
        self.assertEqual(app.metric[1].value, "8")
        self.assertEqual(app.metric[2].value, "56.8")
        self.assertEqual(app.metric[3].value, "15")
        self.assertIn("Apexloom", app.selectbox[0].value)
        self.assertTrue(app.text_area)
        self.assertIn("HYPOTHETICAL DRAFT", app.text_area[0].value)


def _visible_text(app) -> str:
    chunks = []
    for name in ("markdown", "text", "caption", "metric", "expander", "text_area"):
        for item in getattr(app, name):
            chunks.append(str(getattr(item, "value", "")))
            chunks.append(str(getattr(item, "body", "")))
            chunks.append(str(getattr(item, "label", "")))
    return "\n".join(chunks)


if __name__ == "__main__":
    unittest.main()
