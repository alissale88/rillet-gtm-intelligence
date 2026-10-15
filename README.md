# Rillet GTM Intelligence — Account Prioritization Engine

Independent work sample for a GTM Associate interview. It is a local dashboard that turns a company list into a prioritized research queue: who to look at first, why they might be a fit, and what a first note could say.

This is **not** a Rillet product, and the scoring weights are **not** Rillet's ICP. They are explicit hypotheses about where an AI-native ERP might reduce manual finance work. The 30 companies in the sample file are fictional.

The app answers three questions:

1. Which companies should we prioritize for outreach?
2. Why might they be a fit for an AI-native ERP?
3. What messaging should we test, without claiming a problem we have not verified?

## 3-minute demo

1. Read the disclaimer at the top.
2. Scan the four summary cards and **Growth Team Recommendations**. Those sentences are calculated from the loaded file.
3. Open **Apexloom** (first row). Walk the score breakdown, the suggested buyer, and the hypothetical email.
4. Open **Lumengrid Systems** to show a high score that still misses the headcount rule (1,200 employees).
5. Open **Pinnacle Forge** to show a zero score and a draft that says not to send yet.
6. Download the prioritized CSV, or mention that a growth team can upload its own list.

## Install

You need Python 3.10 or newer. Use a virtual environment so an older system copy of NumPy does not get in the way of the tables. From this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate` instead of `source`.

## Run the app

```bash
streamlit run app.py
```

Streamlit prints a local address, usually [http://localhost:8501](http://localhost:8501). Open that in a browser. Leave the terminal window open while you present. Stop the app with Ctrl+C.

If that port is already in use:

```bash
streamlit run app.py --server.port 8502
```

No API keys, database, or network services are required.

## Run the tests

```bash
python -m unittest test_scoring.py
```

The tests check that scores stay between 0 and 100, priority bands are correct, missing cells do not crash the run, high-priority accounts sort first, exports use the right columns, and outreach copy follows the signals that actually scored.

## How scoring works

The rules live in `scoring.py` in `DEFAULT_WEIGHTS`. Change those numbers, or use **Adjust hypothesis weights** in the sidebar. The displayed score is always capped at 100.

| Signal | Default points | When it scores |
| --- | ---: | --- |
| Complex billing | 30 | The cell is Yes |
| Multiple legal entities | 25 | The cell is Yes |
| Finance team hiring | 20 | The cell is Yes |
| Headcount | 15 | Employee count is 50 through 1,000, inclusive |
| Growth stage | 10 | Funding stage is Series A, B, C, or D |

| Score | Priority |
| --- | --- |
| 80–100 | HIGH PRIORITY |
| 50–79 | MEDIUM PRIORITY |
| 0–49 | LOW PRIORITY |

A blank cell adds zero points. A blank is not treated as a No. Unreadable values (for example `Employee Count = many`) also add zero points, and the sidebar lists a warning instead of failing the upload.

With the default weights, high priority requires **both** complex billing and multiple legal entities, plus enough of the other signals to reach 80. That is a property of these weights, not a finding about the market. The methodology panel in the app says so, and it updates if you change the weights.

Headcount and funding stage change the score. They are not treated as buying signals in the email. Outreach copy uses the two strongest of these operational signals:

- Complex billing → revenue recognition and reducing manual accounting workflows
- Multiple legal entities → financial consolidation and multi-entity reporting
- Finance team hiring → scaling finance operations efficiently

If none of those are flagged Yes, the draft says not to send yet.

Suggested buyer, from size and stage only:

- **CFO** at 500+ employees, or Series D / Series E / Public
- **VP Finance** at 150+ employees, or Series B / Series C
- **Controller** otherwise

That persona is a hypothesis, not a verified contact. The draft does not invent a person's name.

### Worked example

**Northstar Ledger** — B2B SaaS, 180 employees, Series B, usage-based, all three operational flags Yes.

- +30 complex billing
- +25 multiple legal entities
- +20 finance team hiring
- +15 headcount is inside 50–1,000
- +10 Series B

**Score: 100 / 100, HIGH PRIORITY.** Suggested buyer: VP Finance, because the company is mid-market and Series B, not because a VP Finance was found. The email leads with revenue recognition and multi-entity consolidation. Finance hiring still adds points, but it is the third signal, so it stays out of the note.

Several sample accounts also score 100. Ties break A to Z, so **Apexloom** appears first.

## What the dashboard shows

- Summary cards: accounts analyzed, high-priority count, average score, complex-billing count
- Growth Team Recommendations: high-priority count, most common buying signals, the industry with the highest average score, and three next actions
- A sortable table, plus search and filters for industry, priority, and funding stage
- An account review with the score breakdown, why the account ranked there, a buyer hypothesis, an outreach angle, and a hypothetical email of about 75 words
- CSV upload, a column template, and a download of the prioritized list

Filters change the table. The recommendation block always describes the full loaded file.

## CSV upload and export

Required upload columns:

`Company Name`, `Industry`, `Employee Count`, `Funding Stage`, `Business Model`, `Complex Billing`, `Multiple Legal Entities`, `Finance Team Hiring`

Optional: `Website`, `Notes`. A `Data Label` column is optional. If every row says `SYNTHETIC DEMO DATA`, the app keeps the synthetic-data banner.

Yes/No cells also accept `Y/N`, `true/false`, and `1/0`. Headers are matched without caring about case. A few aliases work too, such as `Employees`, `Headcount`, and `Funding`.

The downloaded file is the research queue, not a round-trip of the original file. Columns:

`Company Name`, `Industry`, `ICP Score`, `Priority`, `Score Explanation`, `Recommended Buyer`, `Outreach Angle`, `Draft Email`

A bad file does not take down the demo. Missing columns, a broken CSV, or a file with no company names shows an error, and the sample list stays on screen.

## Synthetic versus verified

| Synthetic or hypothetical | Not in this app |
| --- | --- |
| The 30 company names, websites, headcount, funding, and flags | Rillet's real ICP or scoring model |
| Every point value | Live enrichment, intent data, or technographics |
| Buyer persona and email copy | Verified contacts or permission to send |
| "May indicate" / "could benefit from" language | A claim that a company has an accounting problem |
| | Conversion rates, ROI, pipeline, or revenue impact |

Sample websites use the reserved `.example` domain. Every sample note starts with `SYNTHETIC DEMO DATA`.

## What real GTM data would improve

1. Refit the weights on closed-won and closed-lost opportunities, instead of the interview defaults.
2. Replace the hand-entered flags with verified signals: billing model, entity count, and finance job posts.
3. Add contacts and suppression rules so a draft can be routed to a real person, or held.
4. Test whether high-priority accounts actually convert at a higher rate than the rest of the list. Until that test exists, the score is a research order, not a forecast.

## Project files

| File | Role |
| --- | --- |
| `app.py` | Streamlit dashboard |
| `scoring.py` | Weights, scoring, explanations, outreach, insights, CSV validation |
| `sample_accounts.csv` | 30 fictional companies |
| `test_scoring.py` | Automated checks |
| `requirements.txt` | Python packages |
