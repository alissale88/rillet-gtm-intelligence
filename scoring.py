"""Rule-based ICP scoring for the Rillet GTM Intelligence work sample.

The weights below are interview hypotheses, not Rillet's actual ICP.
Change DEFAULT_WEIGHTS to retune the model. Every displayed score is capped
at 100, and missing values add zero points. A blank is not treated as a No.

Default signals
- Complex billing: +30
- Multiple legal entities: +25
- Finance team hiring: +20
- Employee count from 50 through 1,000: +15
- Funding stage Series A, B, C, or D: +10

Priority
- 80 to 100: HIGH PRIORITY
- 50 to 79: MEDIUM PRIORITY
- 0 to 49: LOW PRIORITY
"""

from __future__ import annotations

import io
import math
from pathlib import Path

import pandas as pd

DEFAULT_WEIGHTS = {
    "complex_billing": 30,
    "multiple_entities": 25,
    "finance_hiring": 20,
    "employee_band": 15,
    "growth_stage": 10,
}

HIGH_PRIORITY_MIN = 80
MEDIUM_PRIORITY_MIN = 50
EMPLOYEE_MIN = 50
EMPLOYEE_MAX = 1000

GROWTH_STAGES = {"series a", "series b", "series c", "series d"}
LATER_STAGES = {"series d", "series e", "series f", "public"}
MID_STAGES = {"series b", "series c"}

REQUIRED_COLUMNS = [
    "Company Name",
    "Industry",
    "Employee Count",
    "Funding Stage",
    "Business Model",
    "Complex Billing",
    "Multiple Legal Entities",
    "Finance Team Hiring",
]

OPTIONAL_COLUMNS = ["Website", "Notes", "Data Label"]

EXPORT_COLUMNS = [
    "Company Name",
    "Industry",
    "ICP Score",
    "Priority",
    "Score Explanation",
    "Recommended Buyer",
    "Outreach Angle",
    "Draft Email",
]

GENERATED_COLUMNS = EXPORT_COLUMNS + [
    "Primary Pain Point",
    "Why It Matters",
    "Email Subject",
    "Outreach Status",
]

# Friendly header aliases. Canonical names also match regardless of case.
COLUMN_ALIASES = {
    "company": "Company Name",
    "account": "Company Name",
    "account name": "Company Name",
    "employees": "Employee Count",
    "employee": "Employee Count",
    "employees count": "Employee Count",
    "headcount": "Employee Count",
    "fte": "Employee Count",
    "stage": "Funding Stage",
    "funding": "Funding Stage",
    "funding round": "Funding Stage",
    "complex billing model": "Complex Billing",
    "billing complexity": "Complex Billing",
    "multiple entities": "Multiple Legal Entities",
    "multi entity": "Multiple Legal Entities",
    "multi-entity": "Multiple Legal Entities",
    "legal entities": "Multiple Legal Entities",
    "finance hiring": "Finance Team Hiring",
    "finance team is hiring": "Finance Team Hiring",
    "sector": "Industry",
    "vertical": "Industry",
    "website url": "Website",
    "domain": "Website",
    "url": "Website",
    "note": "Notes",
    "comment": "Notes",
    "comments": "Notes",
    "business type": "Business Model",
    "data label": "Data Label",
}

TRUE_VALUES = {"yes", "y", "true", "1"}
FALSE_VALUES = {"no", "n", "false", "0"}
BLANK_VALUES = {"", "nan", "none", "null", "n/a", "na", "-", "—"}

STAGE_PRETTY = {
    "pre seed": "Pre-seed",
    "seed": "Seed",
    "series a": "Series A",
    "series b": "Series B",
    "series c": "Series C",
    "series d": "Series D",
    "series e": "Series E",
    "series f": "Series F",
    "public": "Public",
    "bootstrapped": "Bootstrapped",
    "bootstrap": "Bootstrapped",
}

BUYING_SIGNALS = [
    {
        "id": "complex_billing",
        "label": "Complex billing",
        "column": "Complex Billing",
        "angle": "revenue recognition and reducing manual accounting workflows",
        "email_help": "revenue recognition and manual accounting workflows",
        "flag_name": "Complex billing",
        "pain": (
            "Potential revenue recognition complexity and manual accounting work "
            "that can come with a complex billing model."
        ),
    },
    {
        "id": "multiple_entities",
        "label": "Multiple legal entities",
        "column": "Multiple Legal Entities",
        "angle": "financial consolidation and multi-entity reporting",
        "email_help": "financial consolidation and multi-entity reporting",
        "flag_name": "Multiple legal entities",
        "pain": "Potential need for financial consolidation and multi-entity reporting.",
    },
    {
        "id": "finance_hiring",
        "label": "Finance team hiring",
        "column": "Finance Team Hiring",
        "angle": "scaling finance operations efficiently",
        "email_help": "scaling finance operations efficiently",
        "flag_name": "Finance team hiring",
        "pain": "Potential need to scale finance operations while the team is hiring.",
    },
]


class DataValidationError(Exception):
    """Raised when a CSV cannot be scored. The message is safe to show in the app."""


def priority_for(score: float) -> str:
    if score >= HIGH_PRIORITY_MIN:
        return "HIGH PRIORITY"
    if score >= MEDIUM_PRIORITY_MIN:
        return "MEDIUM PRIORITY"
    return "LOW PRIORITY"


def resolve_weights(weights: dict | None = None) -> dict:
    """Return a complete weight map. Unknown keys are ignored. Negatives become 0."""
    resolved = dict(DEFAULT_WEIGHTS)
    if weights:
        for key in DEFAULT_WEIGHTS:
            if key in weights and weights[key] is not None:
                resolved[key] = weights[key]
    clean = {}
    for key, value in resolved.items():
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise DataValidationError(f"Weight '{key}' must be a number.") from exc
        if math.isnan(number):
            raise DataValidationError(f"Weight '{key}' must be a number.")
        number = max(0, number)
        if abs(number - round(number)) < 1e-9:
            number = int(round(number))
        clean[key] = number
    return clean


def format_avg(value: float) -> str:
    """One decimal place, with halves rounded away from zero (71.25 -> 71.3)."""
    scaled = value * 10
    if scaled >= 0:
        rounded = math.floor(scaled + 0.5)
    else:
        rounded = math.ceil(scaled - 0.5)
    return f"{rounded / 10:.1f}"


def format_points(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.1f}"


def _clean_header(value: object) -> str:
    text = str(value).replace("\ufeff", "").strip()
    return " ".join(text.split())


def _header_key(value: object) -> str:
    return _clean_header(value).casefold()


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    canonical = {column.casefold(): column for column in REQUIRED_COLUMNS + OPTIONAL_COLUMNS}
    canonical.update(COLUMN_ALIASES)
    new_names = []
    seen: dict[str, str] = {}
    for column in df.columns:
        key = _header_key(column)
        name = canonical.get(key, _clean_header(column))
        if name in seen:
            raise DataValidationError(
                f"The CSV has two columns that both map to '{name}' "
                f"('{seen[name]}' and '{_clean_header(column)}'). Keep one column per field."
            )
        seen[name] = _clean_header(column)
        new_names.append(name)
    renamed = df.copy()
    renamed.columns = new_names
    return renamed


def _is_blank(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    try:
        if pd.isna(value):
            return True
    except TypeError:
        pass
    return str(value).strip().casefold() in BLANK_VALUES


def parse_flag(value: object) -> str:
    """Return yes, no, missing, or unreadable."""
    if isinstance(value, str) and value.strip().casefold() == "unreadable":
        return "unreadable"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if _is_blank(value):
        return "missing"
    text = str(value).strip().casefold()
    if text in TRUE_VALUES:
        return "yes"
    if text in FALSE_VALUES:
        return "no"
    return "unreadable"


def flag_display(status: str) -> str:
    return {"yes": "Yes", "no": "No", "unreadable": "Unreadable"}.get(status, "")


def parse_employees(value: object) -> tuple[object, str]:
    """Return (int | 'Unreadable' | None, status).

    status is set, missing, or unreadable.
    """
    if isinstance(value, str) and value.strip().casefold() == "unreadable":
        return "Unreadable", "unreadable"
    if isinstance(value, bool) or _is_blank(value):
        if isinstance(value, bool):
            return "Unreadable", "unreadable"
        return None, "missing"
    if isinstance(value, int) and not isinstance(value, bool):
        if value < 0:
            return "Unreadable", "unreadable"
        return int(value), "set"
    if isinstance(value, float):
        if value < 0 or abs(value - round(value)) > 1e-9:
            return "Unreadable", "unreadable"
        return int(round(value)), "set"
    text = str(value).strip().replace(",", "")
    if text.endswith(".0"):
        text = text[:-2]
    if text.isdigit():
        return int(text), "set"
    return "Unreadable", "unreadable"


def canonicalize_stage(value: object) -> str:
    if _is_blank(value):
        return ""
    text = str(value).strip().replace("–", "-").replace("—", "-")
    text = " ".join(text.replace("-", " ").split())
    return STAGE_PRETTY.get(text.casefold(), text)


def is_growth_stage(stage: str) -> bool:
    return stage.casefold() in GROWTH_STAGES


def _text_cell(value: object) -> str:
    if _is_blank(value):
        return ""
    return " ".join(str(value).split())


def read_company_csv(source: object) -> pd.DataFrame:
    """Read a CSV from a path, bytes, or file object. Raises DataValidationError."""
    if isinstance(source, pd.DataFrame):
        return source.copy()
    try:
        raw = _source_bytes(source)
    except DataValidationError:
        raise
    except Exception as exc:
        raise DataValidationError(
            "This file could not be read as a CSV. Confirm the format and try again."
        ) from exc
    return _parse_csv_bytes(raw)


def _source_bytes(source: object) -> bytes:
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise DataValidationError(f"Could not find the file '{path.name}'.")
        return path.read_bytes()
    if isinstance(source, bytes):
        return source
    if hasattr(source, "getvalue"):
        data = source.getvalue()
        if isinstance(data, str):
            return data.encode("utf-8")
        return data
    if hasattr(source, "read"):
        data = source.read()
        if isinstance(data, str):
            return data.encode("utf-8")
        return data
    raise DataValidationError("This file could not be read as a CSV.")


def _parse_csv_bytes(raw: bytes) -> pd.DataFrame:
    if raw is None or not raw.strip():
        raise DataValidationError(
            "This file is empty. Upload a CSV with a header row and one row per company."
        )
    text = _decode_csv(raw)
    try:
        frame = pd.read_csv(io.StringIO(text))
    except pd.errors.EmptyDataError as exc:
        raise DataValidationError(
            "This file is empty. Upload a CSV with a header row and one row per company."
        ) from exc
    except pd.errors.ParserError as exc:
        raise DataValidationError(
            "This file could not be parsed as a CSV. Check for broken quotes or extra commas and try again."
        ) from exc
    separator = _alternate_separator(frame)
    if separator:
        try:
            frame = pd.read_csv(io.StringIO(text), sep=separator)
        except pd.errors.ParserError as exc:
            raise DataValidationError(
                "This file could not be parsed as a CSV. Check the delimiter and try again."
            ) from exc
    if frame.shape[1] == 0:
        raise DataValidationError("This CSV has no columns.")
    return frame


def _decode_csv(raw: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp1252"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise DataValidationError(
        "This file could not be read as text. Save the CSV as UTF-8 and try again."
    )


def _alternate_separator(frame: pd.DataFrame) -> str | None:
    if len(frame.columns) != 1:
        return None
    header = str(frame.columns[0])
    if header.count(";") >= 2:
        return ";"
    if header.count("\t") >= 2:
        return "\t"
    return None


def normalize_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Validate columns and clean one row per named company.

    Returns the cleaned frame and human-readable warnings. Scoring does not
    raise for a bad cell; that cell simply adds no points.
    """
    if not isinstance(df, pd.DataFrame):
        raise DataValidationError("The uploaded data is not a table.")
    working = rename_columns(df)
    drop_cols = [column for column in GENERATED_COLUMNS if column in working.columns and column not in REQUIRED_COLUMNS]
    if drop_cols:
        working = working.drop(columns=drop_cols)
    missing = [column for column in REQUIRED_COLUMNS if column not in working.columns]
    if missing:
        raise DataValidationError(
            "The CSV is missing required columns: "
            + ", ".join(missing)
            + ". Required columns are: "
            + ", ".join(REQUIRED_COLUMNS)
            + "."
        )
    if working.empty:
        raise DataValidationError(
            "The CSV has column headers but no company rows. Add one row per company and try again."
        )

    warnings: list[str] = []
    records = []
    for position, (_, row) in enumerate(working.iterrows()):
        row_number = position + 2
        company = _text_cell(row.get("Company Name"))
        if not company:
            warnings.append(f"Row {row_number} was skipped because Company Name is empty.")
            continue

        record = {"Company Name": company}
        industry = _text_cell(row.get("Industry"))
        record["Industry"] = industry or "Not provided"
        if not industry:
            warnings.append(f"Row {row_number} ({company}): Industry is blank.")

        employees, employee_status = parse_employees(row.get("Employee Count"))
        record["Employee Count"] = employees
        if employee_status == "unreadable":
            warnings.append(
                f"Row {row_number} ({company}): Employee Count value "
                f"'{_preview(row.get('Employee Count'))}' was not recognized and did not score."
            )
        elif employee_status == "missing":
            warnings.append(f"Row {row_number} ({company}): Employee Count is blank and did not score.")

        record["Funding Stage"] = canonicalize_stage(row.get("Funding Stage"))
        if not record["Funding Stage"]:
            warnings.append(f"Row {row_number} ({company}): Funding Stage is blank and did not score.")

        record["Business Model"] = _text_cell(row.get("Business Model"))
        if not record["Business Model"]:
            warnings.append(f"Row {row_number} ({company}): Business Model is blank.")

        for column in ("Complex Billing", "Multiple Legal Entities", "Finance Team Hiring"):
            status = parse_flag(row.get(column))
            record[column] = flag_display(status)
            if status == "unreadable":
                warnings.append(
                    f"Row {row_number} ({company}): {column} value "
                    f"'{_preview(row.get(column))}' was not recognized and did not score."
                )
            elif status == "missing":
                warnings.append(
                    f"Row {row_number} ({company}): {column} is blank and did not score. A blank is not treated as No."
                )

        record["Website"] = _text_cell(row.get("Website")) if "Website" in working.columns else ""
        record["Notes"] = _text_cell(row.get("Notes")) if "Notes" in working.columns else ""
        if "Data Label" in working.columns:
            record["Data Label"] = _text_cell(row.get("Data Label"))
        records.append(record)

    if not records:
        raise DataValidationError("No usable rows. Each row needs a company name.")
    return pd.DataFrame.from_records(records), warnings


def _preview(value: object) -> str:
    text = " ".join(str(value).split())
    if len(text) > 80:
        return text[:77] + "..."
    return text


def score_dataframe(df: pd.DataFrame, weights: dict | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Score every company and return a frame sorted from highest score to lowest."""
    resolved = resolve_weights(weights)
    cleaned, warnings = normalize_frame(df)
    records = []
    for _, row in cleaned.iterrows():
        evaluation = evaluate(row.to_dict(), resolved)
        combined = row.to_dict()
        combined.update(evaluation["columns"])
        records.append(combined)
    scored = pd.DataFrame.from_records(records)
    return rank_accounts(scored), warnings


def rank_accounts(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.reset_index(drop=True)
    ranked = df.copy()
    ranked["_name_sort"] = ranked["Company Name"].fillna("").astype(str).str.casefold()
    ranked = ranked.sort_values(
        ["ICP Score", "_name_sort"],
        ascending=[False, True],
        kind="mergesort",
    ).drop(columns=["_name_sort"])
    return ranked.reset_index(drop=True)


def evaluate(row: dict, weights: dict | None = None) -> dict:
    """Score one normalized or raw company row and build the outreach copy."""
    resolved = resolve_weights(weights)
    company = _text_cell(row.get("Company Name")) or "This company"
    industry = _text_cell(row.get("Industry")) or "Not provided"
    business_model = _text_cell(row.get("Business Model"))
    stage = canonicalize_stage(row.get("Funding Stage"))
    employees, _employee_status = parse_employees(row.get("Employee Count"))

    components = [
        _billing_component(row, resolved, business_model),
        _entity_component(row, resolved),
        _hiring_component(row, resolved),
        _employee_component(employees, resolved),
        _stage_component(stage, resolved),
    ]
    raw_points = sum(component["points"] for component in components)
    score = int(round(raw_points))
    capped = False
    if score > 100:
        score = 100
        capped = True
    if score < 0:
        score = 0
    priority = priority_for(score)

    positives = [component for component in components if component["points"] > 0]
    positives.sort(key=lambda component: (-component["points"], component["order"]))
    operational = [component for component in positives if component["kind"] == "buying"]
    top_buying = operational[0]["label"] if operational else "None identified"

    explanation = _explanation_text(score, priority, components, raw_points, capped)
    why = _why_it_matters(company, score, priority, positives, capped, raw_points)
    buyer, buyer_rationale = recommend_buyer(employees, stage)
    pain = operational[0]["pain"] if operational else (
        "No primary pain point is supported by the hypothesized signals. "
        "Outreach should stay neutral until research confirms a trigger."
    )
    angle = outreach_angle(operational)
    subject, body = build_email(
        company=company,
        industry=industry,
        persona=buyer,
        stage=stage,
        employees=employees if isinstance(employees, int) else None,
        operational=operational,
    )
    status = outreach_status(operational)
    draft = (
        f"HYPOTHETICAL DRAFT\n"
        f"Status: {status}\n"
        f"Subject: {subject}\n\n"
        f"{body}\n\n"
        f"Best,\n[Your name]"
    )
    return {
        "columns": {
            "ICP Score": score,
            "Priority": priority,
            "Top Buying Signal": top_buying,
            "Score Explanation": explanation,
            "Recommended Buyer": buyer,
            "Primary Pain Point": pain,
            "Outreach Angle": angle,
            "Draft Email": draft,
            "Why It Matters": why,
            "Email Subject": subject,
            "Outreach Status": status,
        },
        "components": components,
        "positives": positives,
        "operational": operational,
        "subject": subject,
        "email_body": body,
        "why": why,
        "buyer_rationale": buyer_rationale,
        "raw_points": raw_points,
        "capped": capped,
        "industry": industry,
        "stage": stage,
        "employees": employees,
        "business_model": business_model,
    }


def _billing_component(row: dict, weights: dict, business_model: str) -> dict:
    status = parse_flag(row.get("Complex Billing"))
    points = weights["complex_billing"] if status == "yes" else 0
    if status == "yes":
        if business_model:
            reason = (
                f"Complex billing is flagged and the listed business model is {business_model}. "
                "That may indicate potential revenue recognition complexity and manual accounting workflows."
            )
        else:
            reason = (
                "Complex billing is flagged. That may indicate potential revenue recognition "
                "complexity and manual accounting workflows."
            )
    elif status == "no":
        reason = "Complex billing is flagged No, so this signal added no points."
    elif status == "unreadable":
        reason = "Complex billing could not be read, so this signal added no points."
    else:
        reason = "Complex billing is blank, so this signal added no points. A blank is not treated as a No."
    return _component("complex_billing", "Complex billing", points, weights["complex_billing"], reason, "buying", 0, BUYING_SIGNALS[0])


def _entity_component(row: dict, weights: dict) -> dict:
    status = parse_flag(row.get("Multiple Legal Entities"))
    points = weights["multiple_entities"] if status == "yes" else 0
    if status == "yes":
        reason = (
            "Multiple legal entities are flagged. That may increase the potential need "
            "for financial consolidation and multi-entity reporting."
        )
    elif status == "no":
        reason = "Multiple legal entities are flagged No, so this signal added no points."
    elif status == "unreadable":
        reason = "Multiple legal entities could not be read, so this signal added no points."
    else:
        reason = (
            "The multiple-entities field is blank, so this signal added no points. A blank is not treated as a No."
        )
    return _component(
        "multiple_entities",
        "Multiple legal entities",
        points,
        weights["multiple_entities"],
        reason,
        "buying",
        1,
        BUYING_SIGNALS[1],
    )


def _hiring_component(row: dict, weights: dict) -> dict:
    status = parse_flag(row.get("Finance Team Hiring"))
    points = weights["finance_hiring"] if status == "yes" else 0
    if status == "yes":
        reason = (
            "Finance team hiring is flagged. That may indicate investment in financial operations "
            "and a potential need to scale them efficiently."
        )
    elif status == "no":
        reason = "Finance team hiring is flagged No, so this signal added no points."
    elif status == "unreadable":
        reason = "Finance team hiring could not be read, so this signal added no points."
    else:
        reason = "Finance team hiring is blank, so this signal added no points. A blank is not treated as a No."
    return _component(
        "finance_hiring",
        "Finance team hiring",
        points,
        weights["finance_hiring"],
        reason,
        "buying",
        2,
        BUYING_SIGNALS[2],
    )


def _employee_component(employees: object, weights: dict) -> dict:
    points = 0
    if employees == "Unreadable":
        reason = "Employee count could not be read, so the size rule added no points."
    elif not isinstance(employees, int):
        reason = "Employee count is blank, so the size rule added no points."
    elif EMPLOYEE_MIN <= employees <= EMPLOYEE_MAX:
        points = weights["employee_band"]
        reason = f"Employee count ({employees:,}) is inside the hypothesized 50–1,000 range."
    else:
        reason = f"Employee count ({employees:,}) is outside the hypothesized 50–1,000 range."
    return _component("employee_band", "Employee count in range", points, weights["employee_band"], reason, "firmographic", 3, None)


def _stage_component(stage: str, weights: dict) -> dict:
    points = 0
    if not stage:
        reason = "Funding stage is blank, so the growth-stage rule added no points."
    elif is_growth_stage(stage):
        points = weights["growth_stage"]
        reason = f"Funding stage ({stage}) is inside the hypothesized Series A–D range."
    else:
        reason = f"Funding stage ({stage}) is outside the hypothesized Series A–D range."
    return _component("growth_stage", "Growth-stage funding", points, weights["growth_stage"], reason, "firmographic", 4, None)


def _component(signal_id, label, points, max_points, reason, kind, order, catalog) -> dict:
    return {
        "id": signal_id,
        "label": label,
        "points": points,
        "max_points": max_points,
        "explanation": reason,
        "kind": kind,
        "order": order,
        "awarded": points > 0,
        "angle": None if catalog is None else catalog["angle"],
        "email_help": None if catalog is None else catalog["email_help"],
        "flag_name": None if catalog is None else catalog["flag_name"],
        "pain": None if catalog is None else catalog["pain"],
    }


def _explanation_text(score, priority, components, raw_points, capped) -> str:
    parts = [f"Score {score}/100 ({priority})."]
    if capped:
        parts.append(
            f"Raw points are {format_points(raw_points)}, shown as {score} because the score is capped at 100."
        )
    for component in components:
        prefix = f"+{format_points(component['points'])}" if component["points"] else "0"
        parts.append(f"{prefix}: {component['explanation']}")
    parts.append("Hypothetical model only. This does not confirm an accounting issue at the company.")
    return " ".join(parts)


def _why_it_matters(company, score, priority, positives, capped, raw_points) -> str:
    if priority == "HIGH PRIORITY":
        opener = (
            f"{company} scores {score}/100 and is high priority because several hypothesized fit signals are present together."
        )
    elif priority == "MEDIUM PRIORITY":
        opener = (
            f"{company} scores {score}/100 and is medium priority. "
            "Some hypothesized fit signals are present, but not enough to clear 80."
        )
    else:
        opener = (
            f"{company} scores {score}/100 and is low priority. "
            "The available fields show few of the hypothesized fit signals."
        )
    reasons = " ".join(component["explanation"] for component in positives) if positives else "None of the five scoring signals added points."
    caveat = "This explains the score only. It does not confirm an accounting issue at the company."
    cap = ""
    if capped:
        cap = f" Raw points were {format_points(raw_points)} and the displayed score is capped at 100."
    return f"{opener} {reasons}{cap} {caveat}"


def recommend_buyer(employees: object, stage: str) -> tuple[str, str]:
    """Suggest a persona from size and stage. This is not a verified contact."""
    later = stage.casefold() in LATER_STAGES
    mid = stage.casefold() in MID_STAGES
    if (isinstance(employees, int) and employees >= 500) or later:
        return (
            "CFO",
            "Suggested because the company is larger or later-stage, where a CFO often owns "
            "consolidation, close, and reporting. This is a hypothesis, not a verified contact.",
        )
    if (isinstance(employees, int) and employees >= 150) or mid:
        return (
            "VP Finance",
            "Suggested because the company is mid-market or growth-stage, where a VP Finance often "
            "owns the accounting workflow. This is a hypothesis, not a verified contact.",
        )
    return (
        "Controller",
        "Suggested because a Controller is usually closest to day-to-day accounting at earlier or "
        "smaller companies. This is a hypothesis, not a verified contact.",
    )


def outreach_angle(operational: list[dict]) -> str:
    if len(operational) >= 2:
        first, second = operational[0], operational[1]
        return (
            "Hypothetical — review and consider sending. "
            f"Lead with messaging around {first['angle']}, and pair it with {second['angle']}. "
            "These are the two strongest hypothesized signals on this account. "
            "Further signals still count toward the score, but they stay out of the email so the note stays focused."
        )
    if len(operational) == 1:
        only = operational[0]
        return (
            "Hypothetical — optional, because only one operational signal is flagged. "
            f"Lead with messaging around {only['angle']}. "
            "This is the strongest hypothesized operational signal on this account."
        )
    return (
        "Hypothetical — do not send yet. "
        "Do not lead with a specific accounting pain. "
        "Complex billing, multiple legal entities, and finance hiring are not flagged Yes. "
        "Confirm a trigger in research before outreach."
    )


def outreach_status(operational: list[dict]) -> str:
    if len(operational) >= 2:
        return "Review and consider sending"
    if len(operational) == 1:
        return "Optional — one signal only"
    return "Do not send yet"


def build_email(company, industry, persona, stage, employees, operational) -> tuple[str, str]:
    if industry and industry != "Not provided":
        where = f"a company in {industry}"
    else:
        where = "a company on the working list"
    context_bits = []
    if stage:
        context_bits.append(stage)
    if isinstance(employees, int):
        context_bits.append(f"about {employees:,} employees")
    context = f" ({', '.join(context_bits)})" if context_bits else ""
    subject = f"Finance operations at {company}"

    if operational:
        flag = _flag_clause(operational)
        if len(operational) >= 2:
            first, second = operational[0], operational[1]
            benefit = (
                f"That profile could benefit from help with {first['email_help']}, "
                f"alongside {second['email_help']}."
            )
        else:
            benefit = f"That profile could benefit from help with {operational[0]['email_help']}."
        ask = "Would a short conversation be useful to learn how your finance team handles this today?"
    else:
        flag = "No billing, entity, or finance-hiring signal is flagged."
        benefit = "This is not a recommendation to send yet."
        ask = (
            "If research confirms a signal, the note can be rewritten around that evidence. "
            "I would not claim an accounting issue from this list alone."
        )

    body = (
        f"Hi, I am writing to the {persona} at {company}. "
        f"Our working list shows {where}{context}. "
        f"{flag} {benefit} "
        f"{ask}"
    )
    if word_count(body) < 60:
        body += " I am not assuming a verified gap."
    body += " This hypothetical note uses only list data."
    return subject, body


def _flag_clause(operational: list[dict]) -> str:
    """Grammar-safe sentence for the one or two strongest buying signals, in score order."""
    phrases = []
    for item in operational[:2]:
        if item["id"] == "multiple_entities":
            phrases.append("multiple legal entities")
        elif item["id"] == "finance_hiring":
            phrases.append("finance team hiring")
        else:
            phrases.append("complex billing")
    if len(phrases) == 2:
        return f"{_capitalize_first(phrases[0])} and {phrases[1]} are flagged."
    if phrases[0] == "multiple legal entities":
        return "Multiple legal entities are flagged."
    return f"{_capitalize_first(phrases[0])} is flagged."


def _capitalize_first(text: str) -> str:
    if not text:
        return text
    return text[0].upper() + text[1:]


def _lower_first(text: str) -> str:
    if not text:
        return text
    return text[0].lower() + text[1:]


def word_count(text: str) -> int:
    return len(text.replace("—", " ").replace("–", " ").split())


def dataset_is_synthetic(df: pd.DataFrame) -> bool:
    if "Data Label" not in df.columns or df.empty:
        return False
    labels = df["Data Label"].map(lambda value: "" if _is_blank(value) else str(value).strip().upper())
    return bool(len(labels) == len(df) and (labels == "SYNTHETIC DEMO DATA").all())


def model_notes(weights: dict | None = None) -> list[str]:
    """Explain structural consequences of the current weights. Empty when they do not apply."""
    resolved = resolve_weights(weights)
    billing = resolved["complex_billing"]
    entities = resolved["multiple_entities"]
    rest = resolved["finance_hiring"] + resolved["employee_band"] + resolved["growth_stage"]
    notes = [
        "Points are added only when the list shows the signal. Missing values add zero. A blank is not treated as a No.",
        "Headcount and funding stage change the score, but they are not treated as buying signals in outreach copy.",
        "Outreach copy uses at most the two strongest operational signals: complex billing, multiple entities, and finance hiring.",
    ]
    if billing + rest < HIGH_PRIORITY_MIN and entities + rest < HIGH_PRIORITY_MIN and billing + entities < HIGH_PRIORITY_MIN:
        if billing + entities + rest >= HIGH_PRIORITY_MIN:
            notes.append(
                "With these weights, high priority requires both complex billing and multiple legal entities, "
                "plus enough of the remaining signals to reach 80. Neither billing nor entities can reach 80 alone, "
                "and the other three signals cannot reach 80 without both. This is a property of the weights, not a market finding."
            )
    return notes


def export_frame(df: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in EXPORT_COLUMNS if column not in df.columns]
    if missing:
        raise DataValidationError(
            "The prioritized list is missing export columns: " + ", ".join(missing) + "."
        )
    return df.loc[:, EXPORT_COLUMNS].copy()


def export_csv_text(df: pd.DataFrame) -> str:
    return export_frame(df).to_csv(index=False)


def build_insights(df: pd.DataFrame) -> dict:
    """Calculate the Growth Team Recommendations from the loaded, scored dataset."""
    total = int(len(df))
    if total == 0:
        raise DataValidationError("There are no scored accounts to summarize.")

    high = df[df["Priority"] == "HIGH PRIORITY"]
    medium = df[df["Priority"] == "MEDIUM PRIORITY"]
    low = df[df["Priority"] == "LOW PRIORITY"]
    high_n = int(len(high))
    medium_n = int(len(medium))
    low_n = int(len(low))
    avg_score = float(df["ICP Score"].mean())
    billing_n = int((df["Complex Billing"] == "Yes").sum()) if "Complex Billing" in df.columns else 0

    industry = _industry_summary(df)
    scope_df = high if high_n else df
    scope = "high" if high_n else "all"
    signal_rows = _signal_counts(scope_df)
    most_common = _top_labels(signal_rows)
    gap_rows = _medium_gaps(medium) if medium_n else []
    gap_leaders = _top_labels(gap_rows) if gap_rows else []

    summary_lines = [
        f"{high_n} of {total} accounts are high priority.",
        _signal_summary(signal_rows, most_common, high_n, total, scope),
        (
            f"{industry['top_industry']} has the highest average ICP score "
            f"({format_avg(industry['top_avg'])} across {industry['top_count']} accounts)."
        ),
    ]
    actions = _actions(
        high_n=high_n,
        medium_n=medium_n,
        low_n=low_n,
        total=total,
        industry=industry,
        most_common=most_common,
        signal_rows=signal_rows,
        scope=scope,
        gap_leaders=gap_leaders,
        gap_rows=gap_rows,
    )
    return {
        "total": total,
        "high_n": high_n,
        "medium_n": medium_n,
        "low_n": low_n,
        "avg_score": avg_score,
        "complex_billing_n": billing_n,
        "top_industry": industry["top_industry"],
        "top_industry_avg": industry["top_avg"],
        "top_industry_count": industry["top_count"],
        "industry_rows": industry["rows"],
        "signal_rows": signal_rows,
        "signal_scope": scope,
        "most_common_signal_labels": [row["label"] for row in most_common],
        "most_common_signal_count": most_common[0]["count"] if most_common else 0,
        "medium_gap_labels": [row["label"] for row in gap_leaders],
        "medium_gap_count": gap_leaders[0]["count"] if gap_leaders else 0,
        "summary_lines": summary_lines,
        "actions": actions,
    }


def _industry_summary(df: pd.DataFrame) -> dict:
    grouped = (
        df.groupby("Industry", dropna=False)["ICP Score"]
        .agg(avg="mean", n="size")
        .reset_index()
        .sort_values(["avg", "n", "Industry"], ascending=[False, False, True])
    )
    rows = [
        {"industry": str(record.Industry), "avg": float(record.avg), "count": int(record.n)}
        for record in grouped.itertuples(index=False)
    ]
    top = rows[0]
    return {
        "top_industry": top["industry"],
        "top_avg": top["avg"],
        "top_count": top["count"],
        "rows": rows,
    }


def _signal_counts(df: pd.DataFrame) -> list[dict]:
    rows = []
    for signal in BUYING_SIGNALS:
        count = int((df[signal["column"]] == "Yes").sum()) if signal["column"] in df.columns else 0
        rows.append({"id": signal["id"], "label": signal["label"], "count": count, "angle": signal["angle"]})
    return rows


def _medium_gaps(medium: pd.DataFrame) -> list[dict]:
    """Count medium-priority accounts where each buying signal is not Yes."""
    rows = []
    total = len(medium)
    for signal in BUYING_SIGNALS:
        flagged = int((medium[signal["column"]] == "Yes").sum()) if signal["column"] in medium.columns else 0
        rows.append(
            {
                "id": signal["id"],
                "label": signal["label"],
                "count": total - flagged,
                "angle": signal["angle"],
            }
        )
    return rows


def _top_labels(rows: list[dict]) -> list[dict]:
    if not rows:
        return []
    best = max(row["count"] for row in rows)
    if best <= 0:
        return []
    return [row for row in rows if row["count"] == best]


def _signal_summary(signal_rows, most_common, high_n, total, scope) -> str:
    if not most_common:
        if scope == "high":
            return "No buying signals are flagged among the high-priority accounts."
        return "No buying signals are flagged in this dataset."
    labels = _join_and(
        [row["label"] if index == 0 else _lower_first(row["label"]) for index, row in enumerate(most_common)]
    )
    count = most_common[0]["count"]
    if scope == "high":
        if len(most_common) == 1:
            return (
                f"{labels} is the most common buying signal among high-priority accounts "
                f"({count} of {high_n})."
            )
        return (
            f"{labels} are the most common buying signals among high-priority accounts "
            f"({count} of {high_n})."
        )
    if len(most_common) == 1:
        return (
            f"No accounts are high priority, so signal frequency uses the full list. "
            f"{labels} is the most common buying signal ({count} of {total})."
        )
    return (
        f"No accounts are high priority, so signal frequency uses the full list. "
        f"{labels} are the most common buying signals ({count} of {total})."
    )


def _actions(high_n, medium_n, low_n, total, industry, most_common, signal_rows, scope, gap_leaders, gap_rows) -> list[str]:
    industry_name = industry["top_industry"]
    industry_avg = format_avg(industry["top_avg"])
    industry_count = industry["top_count"]
    if high_n:
        first = (
            f"Prioritize research on the {high_n} high-priority accounts. "
            f"{industry_name} currently has the highest average ICP score "
            f"({industry_avg} across {industry_count} accounts)."
        )
    else:
        first = (
            f"No accounts are high priority under the current weights. "
            f"Start research in {industry_name}, which has the highest average ICP score "
            f"({industry_avg} across {industry_count} accounts)."
        )

    if most_common and scope == "high":
        count = most_common[0]["count"]
        if len(most_common) == 1:
            second = (
                f"Lead high-priority outreach with {most_common[0]['angle']}. "
                f"{most_common[0]['label']} is flagged on {count} of {high_n} high-priority accounts."
            )
        else:
            labels = _join_and([_lower_first(row["label"]) for row in most_common])
            if len(most_common) == 2:
                pair = (
                    f"Lead with {most_common[0]['angle']}, and pair that with {most_common[1]['angle']}."
                )
            else:
                pair = "Cover " + _join_and([row["angle"] for row in most_common]) + "."
            second = (
                f"Treat {labels} as the core story for high-priority accounts. "
                f"They are tied at {count} of {high_n}. {pair}"
            )
    elif most_common:
        labels = _join_and([row["label"].lower() for row in most_common])
        count = most_common[0]["count"]
        second = (
            f"Use {labels} as the research theme ({count} of {total} accounts). "
            "Keep the list in medium and low priority until the scores change."
        )
    else:
        second = (
            "No complex-billing, multi-entity, or finance-hiring flags are present. "
            "Do not write pain-led outreach until research confirms one of those signals."
        )

    if medium_n and gap_leaders:
        labels = _join_and([row["label"].lower() for row in gap_leaders])
        count = gap_leaders[0]["count"]
        third = (
            f"Review the {medium_n} medium-priority accounts before the {low_n} low-priority accounts. "
            f"{count} of {medium_n} medium-priority accounts are not flagged for {labels}. "
            "Confirming that field, without assuming it is true, is the most common buying-signal gap in the middle of the list."
        )
    elif low_n and high_n:
        third = (
            f"After the high-priority accounts, the remaining {low_n} are low priority. "
            "Do not spend outreach time there until a billing, entity, or hiring signal is confirmed."
        )
    elif high_n and not medium_n and not low_n:
        third = (
            "Every loaded account is high priority under these weights. "
            "Sequence research by score, then confirm that both complex billing and multiple entities are really present."
        )
    else:
        third = (
            f"All {total} accounts are below the high-priority line. "
            "Spend the next research block filling blank billing, entity, and hiring fields before writing outreach."
        )
    return [first, second, third]


def _join_and(items: list[str]) -> str:
    items = [item for item in items if item]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + ", and " + items[-1]
