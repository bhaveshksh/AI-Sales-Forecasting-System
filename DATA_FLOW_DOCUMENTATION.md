# 🧠 AI Sales Forecasting System — Complete Data Flow Documentation

> **Project:** AI Sales Forecasting System  
> **Stack:** Python · PostgreSQL · Streamlit · Groq LLM (LLaMA 3.3 70B) · Plotly · Pandas  
> **Author:** Sonal  
> **Last Updated:** March 2026  

---

## 🗺️ High-Level Data Flow Overview

```
[ Raw CSV Datasets ]
        │
        ▼
[ ETL Pipeline — database_setup.py ]
        │  (COPY + JOIN)
        ▼
[ PostgreSQL Database — sales_forecasting schema ]
        │  (master_data table)
        ▼
[ Streamlit App — app.py ]
        │  (Filter → Metrics → Charts)
        ▼
[ Prompt Engineering — final_prompt.txt ]
        │  (Dynamic Variable Injection)
        ▼
[ Groq API — LLaMA 3.3 70B Versatile ]
        │  (AI Inference)
        ▼
[ Business Strategy Report (UI Output) ]
```

---

## 📁 Part 1 — Raw Dataset Layer (`Datasets/` folder)

These are the original raw CSV files that feed the entire system. They come from a real-world retail forecasting competition (Corporación Favorita, Ecuador).

| File | Description | Key Columns |
|------|-------------|-------------|
| `train.csv` | Core sales records (3M+ rows, 121 MB) | `id`, `date`, `store_nbr`, `family`, `sales`, `onpromotion` |
| `stores.csv` | Metadata about each store | `store_nbr`, `city`, `state`, `type`, `cluster` |
| `transactions.csv` | Daily customer count per store | `date`, `store_nbr`, `transactions` |
| `holidays_events.csv` | Ecuador national/local public holidays | `date`, `type`, `locale`, `description`, `transferred` |
| `oil.csv` | Daily crude oil prices (WTI) | `date`, `dcoilwtico` |
| `test.csv` | Test data (for submission) | Same structure as train, no `sales` |
| `sample_submission.csv` | Kaggle submission format | `id`, `sales` |

> **Why this matters:** Each CSV represents a different business dimension. By joining them together, we create a rich, multi-dimensional view of retail sales behavior — this is the foundation of all analytics.

---

## ⚙️ Part 2 — ETL Pipeline (`database_setup.py`)

### What is ETL?
**ETL = Extract, Transform, Load**

The `database_setup.py` script is a one-time setup script. It reads all the CSV files, processes them, and loads them into a PostgreSQL database — structured, indexed, and ready for fast queries.

---

### Step 2.1 — Environment Loading

```python
load_dotenv()
DB_USERNAME = os.getenv("DB_USERNAME", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "admin")
DB_HOST     = os.getenv("DB_HOST", "localhost")
DB_PORT     = os.getenv("DB_PORT", "5432")
DB_NAME     = os.getenv("DB_NAME", "postgres")
```

**Explanation:**  
Instead of hardcoding sensitive credentials in the script, the app reads them from a `.env` file using `python-dotenv`. This keeps secrets safe and makes the project portable across different environments (dev, production, cloud).

---

### Step 2.2 — Schema Creation

```sql
CREATE SCHEMA IF NOT EXISTS sales_forecasting;
```

**Explanation:**  
A PostgreSQL **schema** is like a namespace — a folder inside the database that groups related tables together. Using a dedicated `sales_forecasting` schema keeps these tables separate from any other tables in the `postgres` database, preventing naming conflicts.

---

### Step 2.3 — Drop & Recreate Tables (Clean Slate)

```python
tables = ["master_data", "train", "oil", "stores", "transactions", "holidays"]
for t in tables:
    cur.execute(f"DROP TABLE IF EXISTS sales_forecasting.{t} CASCADE;")
```

**Explanation:**  
Before loading fresh data, all old tables are dropped with `CASCADE` (which also removes dependent views/constraints). This prevents data duplication if the script is re-run.

---

### Step 2.4 — Loading `train.csv` (Ultra-Fast COPY)

```python
cur.execute("""
    CREATE TABLE sales_forecasting.train (
        id BIGINT PRIMARY KEY,
        date DATE,
        store_nbr INT,
        family TEXT,
        sales FLOAT,
        onpromotion INT
    );
""")
with open("Datasets/train.csv", "r") as f:
    cur.copy_expert("COPY sales_forecasting.train FROM STDIN WITH CSV HEADER", f)
```

**Explanation:**  
Instead of using slow `INSERT` statements, the script uses PostgreSQL's native `COPY` command via `copy_expert()`. This bulk-loads millions of rows directly from the CSV file into the table at disk speed — typically 10–100x faster than row-by-row inserts. The `train.csv` alone contains ~3 million rows.

---

### Step 2.5 — Loading `stores.csv`, `transactions.csv`, `holidays_events.csv`

The same `COPY` pattern is repeated for each file with appropriate column definitions:

- **`stores`** — 54 rows, store metadata
- **`transactions`** — ~83K rows, daily footfall per store
- **`holidays`** — 350 rows, Ecuador holiday calendar

---

### Step 2.6 — Special Processing: `oil.csv` (Forward Fill)

```python
df_oil = pd.read_csv("Datasets/oil.csv")
df_oil['dcoilwtico'] = df_oil['dcoilwtico'].ffill()
```

**Explanation:**  
The oil price dataset has **missing values** (no data on weekends/holidays). Before loading into PostgreSQL, Pandas is used to **forward-fill** (`ffill`) those gaps — each missing day gets the last known oil price. This ensures no `NULL` values break the analytics downstream.

---

### Step 2.7 — Master Data Table (The Big JOIN)

This is the most important step. All individual tables are joined into a single, denormalized `master_data` table:

```sql
CREATE TABLE sales_forecasting.master_data AS
SELECT 
    t.date,
    t.store_nbr,
    t.family,
    t.sales,
    t.onpromotion,
    CASE WHEN t.sales = 0 THEN 'zero_sales' ELSE 'normal' END AS sales_flag,
    s.city,
    s.state,
    s.type AS store_type,
    tr.transactions,
    o.dcoilwtico,
    h.type AS holiday_type
FROM sales_forecasting.train t
LEFT JOIN sales_forecasting.stores s      ON t.store_nbr = s.store_nbr
LEFT JOIN sales_forecasting.transactions tr ON t.date = tr.date::DATE AND t.store_nbr = tr.store_nbr
LEFT JOIN sales_forecasting.oil o         ON t.date = o.date::DATE
LEFT JOIN sales_forecasting.holidays h    ON t.date = h.date::DATE;
```

**Explanation of each JOIN:**

| JOIN | Purpose |
|------|---------|
| `train → stores` | Enriches each sale with store location (city, state, type, cluster) |
| `train → transactions` | Adds daily footfall count alongside sales volume |
| `train → oil` | Adds WTI crude oil price on each sales date (external economic factor) |
| `train → holidays` | Flags whether a given date is a public holiday (demand driver) |

The `sales_flag` column is a computed column (`CASE WHEN`) that labels rows as `zero_sales` or `normal` for quick filtering.

> **Result:** A single `master_data` table with ~3 million rows, covering every dimension needed for analysis. This is what the Streamlit app queries.

---

## 🔐 Part 3 — Configuration Layer (`.env` File)

```env
GROQ_API_KEY=gsk_...
DB_USERNAME=postgres
DB_PASSWORD=admin
DB_HOST=localhost
DB_PORT=5432
DB_NAME=postgres
```

**Explanation:**  
The `.env` file stores all sensitive credentials:
- **`GROQ_API_KEY`** — Authenticates calls to the Groq cloud API (LLaMA AI model)
- **`DB_*`** — All PostgreSQL connection parameters

Both `database_setup.py` and `app.py` load this file at startup using `load_dotenv()`. This follows the **12-Factor App** methodology — config is separated from code.

> ⚠️ This file should **never** be committed to Git (add it to `.gitignore`).

---

## 🖥️ Part 4 — Streamlit Application (`app.py`)

The Streamlit app is the front-end and the orchestration layer. It connects to PostgreSQL, computes analytics, builds charts, and sends data to the AI.

---

### Step 4.1 — Environment & Config Loading

```python
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DB_USER      = os.getenv("DB_USERNAME", "postgres")
DB_PASS      = os.getenv("DB_PASSWORD", "admin")
DB_HOST      = os.getenv("DB_HOST", "localhost")
DB_PORT      = os.getenv("DB_PORT", "5432")
DB_NAME      = os.getenv("DB_NAME", "postgres")
```

**Explanation:**  
Same `.env` loading as before. The app retrieves both the database credentials and the Groq API key.

---

### Step 4.2 — Database Connection & Data Load

```python
@st.cache_data
def load_data():
    conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST, port=DB_PORT)
    query = "SELECT * FROM sales_forecasting.master_data"
    df = pd.read_sql(query, conn)
    df['date'] = pd.to_datetime(df['date'])
    conn.close()
    return df
```

**Explanation:**  
- `psycopg2.connect()` opens a live connection to PostgreSQL
- `pd.read_sql()` executes the SELECT query and loads the full `master_data` table into a Pandas DataFrame
- `pd.to_datetime()` ensures the `date` column is treated as a datetime object (not a string) — required for time series filtering
- `@st.cache_data` caches the result in memory — the heavy query runs **only once** per session, making the UI fast on every subsequent interaction

---

### Step 4.3 — Sidebar Filters (User Input)

```python
selected_store = st.selectbox("Select Store Number", stores)
date_range     = st.date_input("Select Analysis Period", [...])
```

**Explanation:**  
The sidebar provides two controls:
1. **Store Selector** — The user picks any of the 54 stores. This filters the ~3M row DataFrame down to that store's records.
2. **Date Range Picker** — The user selects a start and end date. Defaults to the last 90 days of available data for the selected store.

These inputs drive ALL downstream analytics dynamically.

---

### Step 4.4 — Analytics Computation

```python
# Filter for selected store & period
mask       = (store_data['date'] >= start_date) & (store_data['date'] <= end_date)
df_period  = store_data.loc[mask].groupby('date')[['sales', 'onpromotion', 'transactions']].sum().reset_index()

# Previous period for comparison
prev_start  = start_date - period_length
df_prev     = store_data.loc[prev_mask].groupby('date')[['sales']].sum().reset_index()

# KPI Calculations
total_sales       = df_period['sales'].sum()
growth_pct        = ((total_sales - prev_sales) / prev_sales) * 100
avg_transactions  = df_period['transactions'].mean()
total_promotions  = df_period['onpromotion'].sum()
```

**Explanation:**

| Metric | How Calculated | Business Meaning |
|--------|---------------|-----------------|
| **Total Sales** | `df_period['sales'].sum()` | Revenue generated in the selected period |
| **Growth %** | `((current - previous) / previous) * 100` | Period-over-period revenue growth |
| **Avg Transactions** | `df_period['transactions'].mean()` | Average daily customer footfall |
| **Total Promotions** | `df_period['onpromotion'].sum()` | Total promotional items during period |
| **Trend** | `"increasing" / "decreasing" / "stable"` | Direction of sales movement |
| **Promo Impact** | Promotions vs avg daily sales | Whether promotions had a significant effect |

---

### Step 4.5 — KPI Metric Cards (UI Display)

```python
c1, c2, c3, c4 = st.columns(4)
st.markdown('<div class="metric-card">...</div>', unsafe_allow_html=True)
```

**Explanation:**  
Four metric cards are rendered at the top of the dashboard in a 4-column layout using custom HTML/CSS:
- 🟢 **Total Sales** — Formatted dollar amount
- 🔵 **Growth %** — With `+/-` sign
- 🟡 **Avg Daily Transactions** — Customer footfall
- 🔴 **Sales Trend** — "Increasing / Decreasing / Stable"

Custom CSS (dark card, green top border, white text) gives a premium analytics dashboard feel.

---

### Step 4.6 — Sales Trajectory Chart

```python
fig = px.line(df_period, x='date', y='sales', line_shape='spline')
fig.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(fig, use_container_width=True)
```

**Explanation:**  
A smooth spline line chart is rendered using **Plotly Express** — the daily aggregated sales for the selected store and date range. The dark theme and transparent background create a polished, professional look. The chart is responsive (full container width).

---

### Step 4.7 — Database Explorer (Expandable)

```python
with st.expander("Explore Postgres Database master_data View"):
    st.dataframe(df_period.head(50))
```

**Explanation:**  
A collapsible section lets users peek at the raw data table directly in the UI — showing up to 50 rows of the filtered period data. This is a transparency/debugging feature.

---

## 🧩 Part 5 — Prompt Engineering Layer (`final_prompt.txt`)

This is the **brain's instruction manual**. The file contains a structured AI prompt with **placeholder variables** in `{curly_braces}` that get filled with real computed values before being sent to the AI.

### Step 5.1 — Prompt Structure

The prompt is divided into:

1. **System Role** — Instructs the AI to think like a senior data analyst + retail strategist + business consultant
2. **Rules** — Data-driven reasoning, no vague answers, always explain WHY
3. **Data Context Block** — Filled with actual computed values from the DataFrame:

```
Sales Summary:
- Time Range: {start_date} to {end_date}
- Total Sales: {total_sales}
- Sales Growth: {growth_percentage}%
- Trend: {increasing/decreasing/stable}

Customer Behavior:
- Avg Transactions: {value}

Promotions:
- Promotion Impact: {high/medium/low}

Forecast Results:
- Next 30/90 Days: {forecast_values}
- Expected Growth: {forecast_growth}%
```

4. **10-Point Analysis Framework** — Instructs the AI to produce:
   - Trend Analysis
   - Root Cause Analysis
   - Customer Behavior Insights
   - Promotion Effectiveness
   - Anomaly Detection
   - Forecast Interpretation
   - Risk Identification
   - Opportunity Identification
   - Strategic Recommendations (Short / Medium / Long term)
   - Decision Support

5. **Output Format** — Tells the AI exactly how to format its response (emojis, bullet points, sections)

---

### Step 5.2 — Dynamic Variable Injection (`app.py`)

```python
filled_prompt = prompt_template \
    .replace("{start_date}",              str(start_date.date())) \
    .replace("{end_date}",                str(end_date.date())) \
    .replace("{total_sales}",             f"${total_sales:,.0f}") \
    .replace("{growth_percentage}",       f"{growth_pct:.1f}") \
    .replace("{increasing/decreasing/stable}", trend) \
    .replace("{value}",                   f"{avg_transactions:,.0f}") \
    .replace("{high/medium/low}",         promo_impact) \
    .replace("{forecast_values}",         forecast_values) \
    .replace("{forecast_growth}",         str(forecast_growth))
```

**Explanation:**  
Python's `.replace()` is called chained on the raw template string. Each placeholder is substituted with the corresponding computed value. The result is a **fully personalized, data-specific prompt** that the Groq LLM receives — different for every store, every date range, every user selection.

> This is the bridge between your database analytics and the AI intelligence layer.

---

## 🤖 Part 6 — AI Inference Layer (Groq API + LLaMA)

### Step 6.1 — API Call

```python
client = Groq(api_key=GROQ_API_KEY)
chat_completion = client.chat.completions.create(
    messages=[{"role": "user", "content": filled_prompt}],
    model="llama-3.3-70b-versatile",
    temperature=0.3,
    max_tokens=1500
)
report = chat_completion.choices[0].message.content
```

**Explanation of each parameter:**

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `model` | `llama-3.3-70b-versatile` | Meta's 70 billion parameter LLaMA 3.3 model — one of the most capable open models |
| `temperature` | `0.3` | Low temperature = more factual, consistent, business-like responses (0 = deterministic, 1 = creative) |
| `max_tokens` | `1500` | Limits response length to ~1500 tokens (~1100 words) — enough for a full strategic report |
| `messages` | `[{role: user, content: filled_prompt}]` | Standard OpenAI-compatible chat format |

**Why Groq?**  
Groq provides blazing-fast LLM inference via their custom LPU (Language Processing Unit) hardware. A 1500-token response is typically generated in **under 3 seconds** — far faster than OpenAI or Azure for interactive use.

---

### Step 6.2 — Response Rendering

```python
st.success("Analysis Complete!")
st.markdown(report)
```

**Explanation:**  
The AI response (a markdown-formatted string) is rendered directly using `st.markdown()`. Streamlit natively parses the emoji-decorated sections, bullet points, and headers — presenting the AI's output as a professional, formatted business report right in the browser.

---

## 📤 Part 7 — Final Output (UI Report)

The final output visible in the Streamlit app consists of:

```
┌─────────────────────────────────────────────────┐
│  📌 Executive Summary                           │
│  📊 Key Insights                                │
│  🔍 Root Cause Analysis                         │
│  👥 Customer Behavior Insights                  │
│  🎯 Promotion Effectiveness                     │
│  🚨 Anomaly Analysis                            │
│  📈 Forecast Insights                           │
│  ⚠️  Risks                                      │
│  🚀 Opportunities                               │
│  💼 Strategic Recommendations                   │
│       Short-Term | Medium-Term | Long-Term      │
│  📢 Final Decision Advice                       │
└─────────────────────────────────────────────────┘
```

This is generated **dynamically** — changing as the user selects different stores or date ranges, triggering a new AI call each time.

---

## 🔄 End-to-End Flow Summary (Step-by-Step)

```
1.  User runs: python database_setup.py
        └─ Reads 7 CSV files from Datasets/
        └─ Creates schema + 5 individual tables in PostgreSQL
        └─ Creates master_data (3M+ rows) via 4-way JOIN
        └─ Done! (runs once)

2.  User runs: streamlit run app.py
        └─ Loads .env (DB creds + API key)
        └─ Connects to PostgreSQL
        └─ Reads all 3M+ rows into Pandas DataFrame (cached)

3.  User selects Store + Date Range in sidebar
        └─ DataFrame filtered to store's records
        └─ Date mask applied
        └─ KPIs computed: total_sales, growth_pct, avg_transactions, trend

4.  Dashboard renders:
        └─ 4 KPI metric cards (custom CSS)
        └─ Plotly spline chart (daily sales trajectory)
        └─ DB expander (raw table preview)

5.  User clicks "Generate AI Strategy Report"
        └─ final_prompt.txt loaded
        └─ All {placeholders} replaced with real computed values
        └─ Filled prompt sent to Groq API
        └─ Model: LLaMA 3.3 70B, temp=0.3, max_tokens=1500

6.  AI Response returned (in ~2-3 seconds)
        └─ Rendered as formatted markdown report
        └─ 10 sections: Summary → KPIs → Root Cause → Decision Support
```

---

## 📦 Tech Stack Summary

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Data Storage | PostgreSQL | Relational database, holds master_data |
| ETL | psycopg2 + Pandas | Fast bulk load via COPY, data cleaning |
| ORM/Query | psycopg2 + SQLAlchemy | DB connection + SQL execution |
| Frontend | Streamlit | Interactive web dashboard |
| Charts | Plotly Express | Interactive time series charts |
| Data Processing | Pandas | DataFrame operations, filtering, aggregation |
| AI Model | LLaMA 3.3 70B (via Groq) | Business intelligence report generation |
| AI Inference | Groq API | Ultra-fast LLM inference (LPU hardware) |
| Config | python-dotenv | Environment variable management |
| Secrets | `.env` file | API keys + DB credentials |

---

## 🗂️ Project File Structure

```
AI Sales Forecasting System/
│
├── .env                            ← Secrets: DB creds + GROQ_API_KEY
├── requirements.txt                ← Python package list
│
├── database_setup.py               ← ETL pipeline (run once)
├── app.py                          ← Streamlit main application
├── final_prompt.txt                ← AI prompt template (with {placeholders})
├── System_prompt.txt               ← AI persona/role definition
│
├── Datasets/                       ← Raw CSV source data
│   ├── train.csv                   ← 3M+ sales records (121 MB)
│   ├── stores.csv                  ← 54 store locations
│   ├── transactions.csv            ← Daily customer counts
│   ├── holidays_events.csv         ← Ecuador holiday calendar
│   ├── oil.csv                     ← WTI crude oil prices
│   ├── test.csv                    ← Test dataset (no labels)
│   └── sample_submission.csv       ← Kaggle format
│
├── ai_sales/                       ← Python virtual environment
│   ├── Scripts/
│   ├── Lib/
│   └── pyvenv.cfg
│
└── DATA_FLOW_DOCUMENTATION.md      ← THIS FILE (you are here 📍)
```

---

## ❓ Key Design Decisions Explained

### Why PostgreSQL instead of just Pandas + CSV?
- The `train.csv` is 121 MB with 3M+ rows — loading it fresh every time would be slow
- PostgreSQL stores it persistently and serves it fast via `psycopg2`
- The 4-way JOIN that creates `master_data` would be complex and slow in pure Pandas
- Future scalability: add indexes, partitioning, user access control easily

### Why `COPY` instead of `INSERT`?
- `COPY` is PostgreSQL's native bulk loader — operates at nearly disk I/O speed
- A regular `INSERT` loop for 3M rows would take minutes; `COPY` does it in seconds

### Why Groq instead of OpenAI?
- **Price:** Groq's free tier is generous for development
- **Speed:** Groq LPU hardware generates tokens 5–10x faster than GPU-based APIs
- **Model Quality:** LLaMA 3.3 70B is competitive with GPT-4o for business analysis tasks

### Why `temperature=0.3`?
- Business reports need **consistency and accuracy**, not creativity
- Low temperature = the model sticks to grounded, factual reasoning
- Higher temperatures (0.7+) would make the AI "hallucinate" data or give inconsistent reports

### Why `@st.cache_data`?
- The `master_data` query returns 3M+ rows — expensive to run repeatedly
- Streamlit re-runs the entire script on every user interaction
- Caching ensures the DB query runs **only once per session**, keeping the UI snappy

---

*Documentation auto-generated and curated based on full source code analysis.*  
*For questions or improvements, refer to the source files: `app.py` and `database_setup.py`.*
