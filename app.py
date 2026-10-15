"""Rillet GTM Intelligence — Account Prioritization Engine.

Local work sample. Run with: streamlit run app.py
Scoring rules live in scoring.py. They are hypotheses, not Rillet's ICP.
"""

from __future__ import annotations

import hashlib
import html
import inspect
import re
from pathlib import Path

import pandas as pd
import streamlit as st

from scoring import (
    DEFAULT_WEIGHTS,
    EXPORT_COLUMNS,
    REQUIRED_COLUMNS,
    build_insights,
    dataset_is_synthetic,
    evaluate,
    export_csv_text,
    format_avg,
    format_points,
    model_notes,
    rank_accounts,
    read_company_csv,
    score_dataframe,
    DataValidationError,
)

ROOT = Path(__file__).resolve().parent
SAMPLE_PATH = ROOT / "sample_accounts.csv"
SAMPLE_LABEL = "Built-in sample · synthetic demo data"

DISPLAY_COLUMNS = [
    "Company Name",
    "Industry",
    "ICP Score",
    "Priority",
    "Top Buying Signal",
    "Recommended Buyer",
]

WEIGHT_FIELDS = [
    ("complex_billing", "Complex billing", "Potential revenue-recognition complexity"),
    ("multiple_entities", "Multiple legal entities", "Potential consolidation and multi-entity reporting"),
    ("finance_hiring", "Finance team hiring", "Potential need to scale finance operations"),
    ("employee_band", "Headcount 50–1,000", "Hypothesized company size band"),
    ("growth_stage", "Series A–D", "Hypothesized growth-stage band"),
]

ABOUT = [
    (
        "Problem",
        "GTM teams can spend significant time manually researching and prioritizing prospects. "
        "The work is repetitive, difficult to explain to sales, and easy to redo from scratch each week.",
    ),
    (
        "Solution",
        "A repeatable scoring and prospecting workflow that turns company information into actionable "
        "recommendations: a score, a priority, the reason it scored, a buyer hypothesis, and a hypothetical outreach note.",
    ),
    (
        "Business Value",
        "Potentially reduces manual account qualification and helps sales teams focus their research on accounts "
        "with the strongest hypothesized fit. This demo does not claim a measured time saving, conversion lift, or pipeline number.",
    ),
    (
        "Limitations",
        "Uses synthetic data and hypothetical scoring criteria. It is not Rillet's actual ICP, and it does not "
        "predict conversion rates, win rates, or revenue.",
    ),
    (
        "Next Steps",
        "Validate ICP weights using Rillet's historical customer and opportunity data, add verified enrichment "
        "signals, and test whether prioritized accounts convert at higher rates.",
    ),
]


def main() -> None:
    st.set_page_config(
        page_title="Rillet GTM Intelligence",
        page_icon="◆",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_css()
    if not SAMPLE_PATH.exists():
        st.error("sample_accounts.csv is missing. Keep it in the same folder as app.py.")
        st.stop()

    _sidebar_intro()
    uploaded = st.sidebar.file_uploader(
        "Upload a company CSV",
        type=["csv"],
        help=(
            "Required columns: "
            + ", ".join(REQUIRED_COLUMNS)
            + ". Website and Notes are optional. Yes/No fields can also be Y/N or true/false."
        ),
    )
    raw, source_label, read_error = _resolve_source(uploaded)
    if read_error:
        st.sidebar.error(read_error)

    weights = _weights_from_state()
    try:
        scored, warnings = score_dataframe(raw, weights)
    except DataValidationError as exc:
        st.sidebar.error(f"{exc} Showing the built-in sample instead.")
        raw = read_company_csv(SAMPLE_PATH)
        source_label = SAMPLE_LABEL
        scored, warnings = score_dataframe(raw, weights)

    _sidebar_followups(uploaded, warnings, source_label)
    query, industries, priorities, stages = _filters(scored)
    _render_weight_editor()
    filtered = apply_filters(scored, query, industries, priorities, stages)

    _hero(source_label, raw)
    insights = build_insights(scored)
    _metrics(insights)
    _legend()
    _recommendations(insights)
    _methodology(weights)
    _accounts(filtered, scored, weights)
    _about()


def _inject_css() -> None:
    st.markdown(
        """
        <style>
          .stApp { background: #F4F1EA; color: #1A2332; }
          .block-container { padding-top: 1.3rem; padding-bottom: 3rem; max-width: 1180px; }
          section[data-testid="stSidebar"] { background: #FBFAF7; border-right: 1px solid #E3DDD2; }
          header[data-testid="stHeader"] { background: transparent; }
          h1, h2, h3 { letter-spacing: -0.03em; }
          .hero-kicker {
            color: #8C6A32; font-size: 0.78rem; font-weight: 700;
            letter-spacing: 0.14em; text-transform: uppercase; margin: 0 0 0.35rem;
          }
          .hero h1 { font-size: 2.15rem; line-height: 1.1; margin: 0; font-weight: 600; color: #1A2332; }
          .lede { color: #3E4C59; font-size: 1.05rem; margin: 0.35rem 0 0.2rem; }
          .questions { color: #5C6B7A; margin: 0 0 0.9rem; }
          .disclaimer {
            background: #FFF; border: 1px solid #E3DDD2; border-left: 4px solid #1C4E4A;
            border-radius: 12px; padding: 0.85rem 1rem; margin: 0 0 1rem; color: #1A2332;
          }
          .disclaimer strong { color: #143835; }
          div[data-testid="stMetric"] {
            background: #FFF; border: 1px solid #E3DDD2; border-radius: 14px; padding: 0.75rem 0.9rem 0.55rem;
          }
          div[data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; }
          .legend { display: flex; flex-wrap: wrap; gap: 0.45rem; margin: 0.2rem 0 0.4rem; }
          .badge {
            display: inline-block; border-radius: 999px; padding: 0.18rem 0.62rem;
            font-size: 0.75rem; font-weight: 700; letter-spacing: 0.04em;
          }
          .badge.high { background: #E7F5EF; color: #0F6E56; }
          .badge.medium { background: #FBF3E2; color: #8A5A10; }
          .badge.low { background: #F1F2F4; color: #5C6773; }
          .badge.neutral { background: #EEF3F2; color: #1C4E4A; }
          .account-head {
            display: flex; justify-content: space-between; gap: 1.5rem; align-items: flex-end; margin-bottom: 0.4rem;
          }
          .account-head h2 { margin: 0.1rem 0 0.25rem; font-size: 1.7rem; }
          .meta { color: #5C6B7A; margin: 0 0 0.55rem; }
          .score-block { min-width: 140px; text-align: right; }
          .score-num { font-size: 2.6rem; font-weight: 700; line-height: 0.9; font-variant-numeric: tabular-nums; }
          .score-den { color: #5C6B7A; margin-bottom: 0.35rem; }
          .score-track, .bar-track { background: #E7E2D8; border-radius: 999px; height: 8px; overflow: hidden; }
          .score-fill, .bar-fill { height: 100%; background: #1C4E4A; border-radius: 999px; }
          .score-fill.medium, .bar-fill.medium { background: #A56B16; }
          .score-fill.low { background: #7B8794; }
          .pills { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-top: 0.15rem; }
          .bar-row { display: grid; grid-template-columns: minmax(140px, 1.1fr) 2fr auto; gap: 0.65rem; align-items: center; margin: 0.38rem 0; }
          .bar-name, .bar-num { font-size: 0.92rem; }
          .bar-num { color: #3E4C59; font-variant-numeric: tabular-nums; white-space: nowrap; }
          .reason { display: flex; gap: 0.75rem; padding: 0.45rem 0; border-bottom: 1px solid #EFEAE2; }
          .pts { min-width: 2.4rem; font-weight: 700; color: #1C4E4A; font-variant-numeric: tabular-nums; }
          .pts.zero { color: #8B95A1; font-weight: 600; }
          .about-kicker { color: #8C6A32; font-size: 0.75rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; margin: 0.85rem 0 0.15rem; }
          .section-note { color: #5C6B7A; margin-top: -0.4rem; }
          @media (max-width: 800px) {
            .account-head { flex-direction: column; align-items: flex-start; }
            .score-block { text-align: left; }
            .bar-row { grid-template-columns: 1fr; gap: 0.2rem; }
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _sidebar_intro() -> None:
    st.sidebar.markdown("### Demo path")
    st.sidebar.caption("About three minutes.")
    st.sidebar.markdown(
        "1. Read the disclaimer. These weights are hypotheses, not Rillet's ICP.\n"
        "2. Scan the summary cards and Growth Team Recommendations.\n"
        "3. Open **Apexloom** and walk the score, the buyer, and the email.\n"
        "4. Contrast **Lumengrid Systems** (high score, headcount miss) and **Pinnacle Forge** (no signal, do not send).\n"
        "5. Download the prioritized list, or upload your own CSV."
    )


def _resolve_source(uploaded) -> tuple[pd.DataFrame, str, str | None]:
    sample = read_company_csv(SAMPLE_PATH)
    if uploaded is None:
        return sample, SAMPLE_LABEL, None

    token = f"{uploaded.name}:{getattr(uploaded, 'size', 0)}"
    if token != st.session_state.get("upload_token"):
        st.session_state["upload_token"] = token
        st.session_state["force_sample"] = False

    if st.session_state.get("force_sample"):
        return sample, SAMPLE_LABEL, None
    try:
        frame = read_company_csv(uploaded)
    except DataValidationError as exc:
        return sample, SAMPLE_LABEL, f"{exc} Showing the built-in sample instead."
    return frame, f"Uploaded CSV · {uploaded.name}", None


def _weights_from_state() -> dict:
    weights = {}
    for key, _label, _help in WEIGHT_FIELDS:
        state_key = f"weight_{key}"
        if state_key not in st.session_state:
            st.session_state[state_key] = int(DEFAULT_WEIGHTS[key])
        value = st.session_state[state_key]
        weights[key] = int(DEFAULT_WEIGHTS[key] if value is None else value)
    return weights


def _render_weight_editor() -> None:
    with st.sidebar.expander("Adjust hypothesis weights", expanded=False):
        st.caption("Defaults sum to 100. Scores still cap at 100. This is not Rillet's ICP.")
        current = []
        for key, label, _help in WEIGHT_FIELDS:
            entered = st.number_input(
                label,
                min_value=0,
                max_value=100,
                step=1,
                key=f"weight_{key}",
            )
            current.append(int(DEFAULT_WEIGHTS[key] if entered is None else entered))
        st.caption(f"Current weights sum to {sum(current)}.")
        _full_width("button", "Reset weights", on_click=_reset_weights)


def _reset_weights() -> None:
    for key, _label, _help in WEIGHT_FIELDS:
        st.session_state[f"weight_{key}"] = int(DEFAULT_WEIGHTS[key])


def _sidebar_followups(uploaded, warnings: list[str], source_label: str) -> None:
    st.sidebar.caption(f"Current data: {source_label}")
    if uploaded is not None and not st.session_state.get("force_sample"):
        if _full_width("button", "Revert to sample data", sidebar=True):
            st.session_state["force_sample"] = True
            st.rerun()
    template = ",".join(REQUIRED_COLUMNS + ["Website", "Notes"]) + "\n"
    _full_width(
        "download",
        "Download CSV template",
        sidebar=True,
        data=template.encode("utf-8-sig"),
        file_name="company_template.csv",
        mime="text/csv",
        key="download_template",
    )
    if warnings:
        with st.sidebar.expander(f"Data warnings ({len(warnings)})", expanded=False):
            for warning in warnings[:40]:
                st.write(warning)
            if len(warnings) > 40:
                st.write(f"{len(warnings) - 40} more warnings are hidden.")


def _filters(scored: pd.DataFrame):
    st.sidebar.markdown("**Filters**")
    query = st.sidebar.text_input("Search companies", placeholder="Company name")
    industries = sorted(scored["Industry"].dropna().astype(str).unique())
    priorities = ["HIGH PRIORITY", "MEDIUM PRIORITY", "LOW PRIORITY"]
    stages = sorted(stage for stage in scored["Funding Stage"].dropna().astype(str).unique() if stage)
    selected_industries = st.sidebar.multiselect("Industry", industries, placeholder="All industries")
    selected_priorities = st.sidebar.multiselect("Priority level", priorities, placeholder="All priorities")
    selected_stages = st.sidebar.multiselect("Funding stage", stages, placeholder="All stages")
    return query, selected_industries, selected_priorities, selected_stages


def apply_filters(df, query, industries, priorities, stages) -> pd.DataFrame:
    filtered = df
    needle = (query or "").strip()
    if needle:
        filtered = filtered[
            filtered["Company Name"].astype(str).str.contains(re.escape(needle), case=False, na=False)
        ]
    if industries:
        filtered = filtered[filtered["Industry"].isin(industries)]
    if priorities:
        filtered = filtered[filtered["Priority"].isin(priorities)]
    if stages:
        filtered = filtered[filtered["Funding Stage"].isin(stages)]
    return rank_accounts(filtered)


def _hero(source_label: str, raw: pd.DataFrame) -> None:
    synthetic = dataset_is_synthetic(raw) or source_label == SAMPLE_LABEL
    if synthetic:
        data_line = "The companies in this view are labeled SYNTHETIC DEMO DATA. They are not real prospects."
    else:
        data_line = "Uploaded rows are scored as provided. This app does not verify them against live sources."
    st.markdown(
        f"""
        <div class="hero">
          <p class="hero-kicker">Independent work sample</p>
          <h1>Rillet GTM Intelligence</h1>
          <p class="lede">Account Prioritization Engine</p>
          <p class="questions">Which accounts should we research first, why might they fit, and what should we say?</p>
        </div>
        <div class="disclaimer">
          <strong>Hypothetical ICP.</strong> These weights are interview hypotheses about where an AI-native ERP
          might reduce manual finance work. They are not Rillet's actual ideal customer profile.
          Messaging is hypothetical and should not be sent as-is. {html.escape(data_line)}
          <br><span class="section-note">Data source: {html.escape(source_label)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _metrics(insights: dict) -> None:
    cards = [
        ("Total accounts analyzed", str(insights["total"]), "Companies scored from the loaded file."),
        ("High-priority accounts", str(insights["high_n"]), "Hypothetical score of 80 to 100."),
        ("Average ICP score", format_avg(insights["avg_score"]), "Mean score. Not a conversion rate."),
        ("Accounts with complex billing", str(insights["complex_billing_n"]), "Rows flagged Yes for complex billing."),
    ]
    columns = st.columns(4)
    for column, (label, value, help_text) in zip(columns, cards):
        column.metric(label, value, help=help_text)


def _legend() -> None:
    st.markdown(
        """
        <div class="legend">
          <span class="badge high">HIGH PRIORITY · 80–100</span>
          <span class="badge medium">MEDIUM PRIORITY · 50–79</span>
          <span class="badge low">LOW PRIORITY · 0–49</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _recommendations(insights: dict) -> None:
    st.markdown("## Growth Team Recommendations")
    st.caption("Calculated from the full loaded dataset. Filters change the table below, not these totals.")
    facts = st.columns(3)
    for column, line in zip(facts, insights["summary_lines"]):
        with column:
            with st.container(border=True):
                st.write(line)

    left, right = st.columns(2)
    with left:
        st.markdown("**Average ICP score by industry**")
        bars = []
        for row in insights["industry_rows"]:
            bars.append(
                _bar(
                    f"{row['industry']} ({row['count']})",
                    format_avg(row["avg"]),
                    row["avg"] / 100,
                )
            )
        st.markdown("".join(bars), unsafe_allow_html=True)
    with right:
        scope = "high-priority accounts" if insights["signal_scope"] == "high" else "all accounts"
        denominator = insights["high_n"] if insights["signal_scope"] == "high" else insights["total"]
        st.markdown(f"**Buying signals among {scope}**")
        bars = []
        for row in insights["signal_rows"]:
            fraction = 0 if denominator == 0 else row["count"] / denominator
            bars.append(_bar(row["label"], f"{row['count']} of {denominator}", fraction))
        st.markdown("".join(bars), unsafe_allow_html=True)

    st.markdown("**Recommended next actions**")
    for index, action in enumerate(insights["actions"], start=1):
        st.write(f"{index}. {action}")


def _bar(label: str, value_label: str, fraction: float) -> str:
    width = max(0.0, min(1.0, float(fraction))) * 100
    return (
        "<div class='bar-row'>"
        f"<div class='bar-name'>{html.escape(label)}</div>"
        f"<div class='bar-track'><div class='bar-fill' style='width:{width:.1f}%'></div></div>"
        f"<div class='bar-num'>{html.escape(value_label)}</div>"
        "</div>"
    )


def _methodology(weights: dict) -> None:
    with st.expander("How this score works", expanded=False):
        st.write(
            "Each signal adds its weight only when the row supports it. The total is capped at 100. "
            "Headcount and funding stage affect the score, but outreach copy uses the operational signals only."
        )
        weight_frame = pd.DataFrame(
            [
                {"Signal": label, "Points": int(weights[key]), "Hypothesis": hint}
                for key, label, hint in WEIGHT_FIELDS
            ]
        )
        _show_dataframe(weight_frame, hide_index=True)
        for note in model_notes(weights):
            st.write(note)
        st.write(
            "Buyer suggestion: CFO at 500+ employees or Series D and later, VP Finance at 150+ employees "
            "or Series B–C, otherwise Controller. That is a hypothesis, not a verified contact."
        )


def _accounts(filtered: pd.DataFrame, scored: pd.DataFrame, weights: dict) -> None:
    st.markdown("## Prioritized accounts")
    st.caption("Sorted from the highest hypothetical ICP score to the lowest. Ties break A to Z.")
    export_all = export_csv_text(scored).encode("utf-8-sig")
    downloads = st.columns(2)
    with downloads[0]:
        _full_width(
            "download",
            "Download prioritized list",
            data=export_all,
            file_name="rillet_gtm_prioritized_accounts.csv",
            mime="text/csv",
            key="download_all",
        )
    if len(filtered) != len(scored):
        with downloads[1]:
            _full_width(
                "download",
                "Download filtered list",
                data=export_csv_text(filtered).encode("utf-8-sig"),
                file_name="rillet_gtm_filtered_accounts.csv",
                mime="text/csv",
                key="download_filtered",
            )
    if filtered.empty:
        st.info("No accounts match the current search and filters.")
        return

    st.caption(f"Showing {len(filtered)} of {len(scored)} accounts.")
    selected_rows = _render_table(filtered)
    labels = [
        f"{row['Company Name']}  ·  {int(row['ICP Score'])}  ·  {row['Priority']}"
        for _, row in filtered.iterrows()
    ]
    if selected_rows:
        picked = labels[selected_rows[0]]
        signature = f"{_table_key(filtered)}:{selected_rows[0]}"
        if st.session_state.get("table_signature") != signature:
            st.session_state["table_signature"] = signature
            st.session_state["company_select"] = picked
    if st.session_state.get("company_select") not in labels:
        st.session_state["company_select"] = labels[0]

    st.selectbox("Review an account", labels, key="company_select")
    chosen = labels.index(st.session_state["company_select"])
    _render_account(filtered.iloc[chosen].to_dict(), weights)


def _render_table(filtered: pd.DataFrame) -> list[int]:
    view = filtered.loc[:, DISPLAY_COLUMNS].copy()
    view["ICP Score"] = view["ICP Score"].astype(int)
    height = min(540, max(168, 42 + 35 * len(view)))
    event = _show_dataframe(
        view,
        hide_index=True,
        height=height,
        on_select="rerun",
        selection_mode="single-row",
        key=_table_key(filtered),
        column_config={
            "Company Name": st.column_config.TextColumn("Company Name"),
            "Industry": st.column_config.TextColumn("Industry"),
            "ICP Score": st.column_config.ProgressColumn(
                "ICP Score",
                help="Hypothetical score from 0 to 100. Not a conversion probability.",
                min_value=0,
                max_value=100,
                format="%d",
            ),
            "Priority": st.column_config.TextColumn("Priority"),
            "Top Buying Signal": st.column_config.TextColumn("Top Buying Signal"),
            "Recommended Buyer": st.column_config.TextColumn("Recommended Buyer"),
        },
    )
    if event is not None and hasattr(event, "selection"):
        return list(event.selection.rows)
    return []


def _table_key(filtered: pd.DataFrame) -> str:
    payload = "|".join(
        f"{name}:{score}" for name, score in zip(filtered["Company Name"], filtered["ICP Score"])
    )
    digest = hashlib.md5(payload.encode("utf-8")).hexdigest()[:12]
    return f"accounts-{digest}"


def _render_account(row: dict, weights: dict) -> None:
    result = evaluate(row, weights)
    columns = result["columns"]
    priority = columns["Priority"]
    tone = {"HIGH PRIORITY": "high", "MEDIUM PRIORITY": "medium"}.get(priority, "low")
    status_tone = "high" if columns["Outreach Status"].startswith("Review") else (
        "medium" if columns["Outreach Status"].startswith("Optional") else "low"
    )
    employees = result["employees"]
    employee_label = f"{employees:,} employees" if isinstance(employees, int) else "Employee count not scored"
    stage = result["stage"] or "Stage not provided"
    model = result["business_model"] or "Business model not provided"
    meta = " · ".join(
        [
            str(result["industry"]),
            stage,
            employee_label,
            model,
            f"Complex billing: {_shown(row.get('Complex Billing'))}",
            f"Multiple entities: {_shown(row.get('Multiple Legal Entities'))}",
            f"Finance hiring: {_shown(row.get('Finance Team Hiring'))}",
        ]
    )
    signal_pills = "".join(
        f"<span class='badge neutral'>{html.escape(item['label'])}</span>" for item in result["operational"]
    ) or "<span class='badge low'>No operational buying signal</span>"
    score = int(columns["ICP Score"])
    st.markdown(
        f"""
        <div class="account-head">
          <div>
            <p class="hero-kicker">Account review</p>
            <h2>{html.escape(str(row.get("Company Name", "")))}</h2>
            <p class="meta">{html.escape(meta)}</p>
            <div class="pills">
              <span class="badge {tone}">{html.escape(priority)}</span>
              <span class="badge {status_tone}">{html.escape(columns["Outreach Status"])}</span>
              {signal_pills}
            </div>
          </div>
          <div class="score-block">
            <div class="score-num">{score}</div>
            <div class="score-den">/ 100</div>
            <div class="score-track"><div class="score-fill {tone}" style="width:{score}%"></div></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    website = str(row.get("Website") or "")
    if website.startswith("http://") or website.startswith("https://"):
        st.markdown(
            f"<a href='{html.escape(website, quote=True)}' target='_blank'>{html.escape(website)}</a>",
            unsafe_allow_html=True,
        )
    notes = str(row.get("Notes") or "").strip()
    if notes:
        st.caption(notes)

    breakdown, outreach = st.columns(2, gap="large")
    with breakdown:
        st.markdown("**Score breakdown**")
        positives = result["positives"]
        zeros = [item for item in result["components"] if item["points"] <= 0]
        if positives:
            st.markdown(_reasons_html(positives), unsafe_allow_html=True)
        else:
            st.write("None of the five signals added points.")
        if result["capped"]:
            st.caption(
                f"Raw points are {format_points(result['raw_points'])}. The displayed score is capped at 100."
            )
        with st.expander("Signals that did not score"):
            if zeros:
                st.markdown(_reasons_html(zeros), unsafe_allow_html=True)
            else:
                st.write("Every signal in the current model added points.")
        st.markdown("**Why this company matters**")
        st.write(result["why"])

    with outreach:
        st.markdown("**Hypothetical outreach**")
        st.caption("Generated from list fields only. Do not present it as a verified diagnosis.")
        st.markdown("**Suggested buyer persona**")
        st.write(columns["Recommended Buyer"])
        st.caption(result["buyer_rationale"])
        st.markdown("**Primary potential pain point**")
        st.write(columns["Primary Pain Point"])
        st.markdown("**Suggested outreach angle**")
        st.write(columns["Outreach Angle"])
        st.markdown("**Cold email draft**")
        st.caption("About 75 words. Hypothetical — edit the role line with a real name before anyone sends it.")
        st.text_area(
            "Hypothetical cold email",
            value=columns["Draft Email"],
            height=320,
            key=f"email-{row.get('Company Name')}-{score}",
            label_visibility="collapsed",
        )


def _reasons_html(components: list[dict]) -> str:
    rows = []
    for component in components:
        points = component["points"]
        label = f"+{format_points(points)}" if points else "0"
        css = "pts" if points else "pts zero"
        rows.append(
            "<div class='reason'>"
            f"<div class='{css}'>{html.escape(label)}</div>"
            f"<div>{html.escape(component['explanation'])}</div>"
            "</div>"
        )
    return "".join(rows)


def _shown(value: object) -> str:
    text = "" if value is None else str(value).strip()
    return text or "Not provided"


def _full_width(kind: str, label: str, sidebar: bool = False, **kwargs):
    """Stretch buttons on current Streamlit, and fall back for older releases."""
    fn = st.button if kind == "button" else st.download_button
    parameters = inspect.signature(fn).parameters
    width_kwargs = {"width": "stretch"} if "width" in parameters else {"use_container_width": True}
    if sidebar:
        fn = st.sidebar.button if kind == "button" else st.sidebar.download_button
    return fn(label, **width_kwargs, **kwargs)


def _show_dataframe(data, **kwargs):
    """Support both the current Streamlit width argument and older releases."""
    parameters = inspect.signature(st.dataframe).parameters
    if "width" in parameters:
        return st.dataframe(data, width="stretch", **kwargs)
    return st.dataframe(data, use_container_width=True, **kwargs)


def _about() -> None:
    st.markdown("## About This Work Sample")
    st.write(
        "Rillet builds an AI-native ERP that helps finance teams automate accounting workflows, "
        "financial close, revenue recognition, and reporting. This dashboard is a personal work sample "
        "for a GTM Associate interview. It is not a Rillet product."
    )
    with st.container(border=True):
        for title, body in ABOUT:
            st.markdown(f"<p class='about-kicker'>{title}</p>", unsafe_allow_html=True)
            st.write(body)


main()
