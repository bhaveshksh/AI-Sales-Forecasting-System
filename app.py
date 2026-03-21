import streamlit as st
import pandas as pd
import plotly.express as px
from groq import Groq
from dotenv import load_dotenv
import os
import psycopg2

# Load Environment Variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DB_USER = os.getenv("DB_USERNAME", "postgres")
DB_PASS = os.getenv("DB_PASSWORD", "admin")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "postgres")

st.set_page_config(page_title="AI Sales Forecaster", page_icon="📈", layout="wide")

st.markdown("""
<style>
.metric-card {
    background-color: #1E1E1E;
    border-radius: 10px;
    padding: 20px;
    margin: 10px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    border-top: 4px solid #4CAF50;
}
.metric-title {
    color: #888;
    font-size: 14px;
    font-weight: 600;
}
.metric-value {
    color: #FFF;
    font-size: 28px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS,
        host=DB_HOST,
        port=DB_PORT
    )
    query = "SELECT * FROM sales_forecasting.master_data"
    df = pd.read_sql(query, conn)
    df['date'] = pd.to_datetime(df['date'])
    conn.close()
    return df

st.title("✨ Sales Forecasting System")
st.markdown("Automated strategic insights using Groq LLM & PostgreSQL Analytics")

try:
    with st.spinner("Loading millions of sales records from PostgreSQL..."):
        df_master = load_data()
except Exception as e:
    st.error(f"Error loading database (please check your .env and ensure Postgres is running & database_setup.py is run):\n{e}")
    st.stop()

stores = sorted(df_master['store_nbr'].unique())

#Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    selected_store = st.selectbox("Select Store Number", stores)
    
    # Filter data for selected store
    store_data = df_master[df_master['store_nbr'] == selected_store]

    # ── Build daily date list: 2013-01-01 → 2017-12-31 ──────────
    all_days   = pd.date_range(start="2013-01-01", end="2017-12-31", freq="D")
    date_labels = [d.strftime("%Y/%m/%d") for d in all_days]   # e.g. "2013/02/11"
    date_map    = {d.strftime("%Y/%m/%d"): d for d in all_days}
    last_label  = "2017/12/31"

    st.markdown("### 📅 Select Analysis Period")

    # ── Quick-select buttons (update session state) ───────────────
    st.markdown("**⚡ Quick Select**")
    q1, q2, q3 = st.columns(3)
    last_date = pd.Timestamp("2017-12-31")
    with q1:
        if st.button("Past 3M", use_container_width=True, help="Past 3 Months"):
            st.session_state["start_date_select"] = (last_date - pd.DateOffset(months=3)).strftime("%Y/%m/%d")
            st.session_state["end_date_select"]   = last_label
    with q2:
        if st.button("Past 6M", use_container_width=True, help="Past 6 Months"):
            st.session_state["start_date_select"] = (last_date - pd.DateOffset(months=6)).strftime("%Y/%m/%d")
            st.session_state["end_date_select"]   = last_label
    with q3:
        if st.button("Past 1Y", use_container_width=True, help="Past 1 Year"):
            st.session_state["start_date_select"] = (last_date - pd.DateOffset(years=1)).strftime("%Y/%m/%d")
            st.session_state["end_date_select"]   = last_label

    # ── Initialise defaults if not yet set ───────────────────────
    if "start_date_select" not in st.session_state:
        st.session_state["start_date_select"] = (last_date - pd.DateOffset(months=3)).strftime("%Y/%m/%d")
    if "end_date_select" not in st.session_state:
        st.session_state["end_date_select"] = last_label

    # ── Start / End date dropdowns ───────────────────────────────
    selected_start_label = st.selectbox(
        "🗓 Start Date",
        options=date_labels,
        index=date_labels.index(st.session_state["start_date_select"]),
        key="start_date_select"
    )
    selected_end_label = st.selectbox(
        "🗓 End Date",
        options=date_labels,
        index=date_labels.index(st.session_state["end_date_select"]),
        key="end_date_select"
    )

    start_date = date_map[selected_start_label]
    end_date   = date_map[selected_end_label]

# Validate order
if start_date >= end_date:
    st.warning("⚠️ Start Date must be before End Date. Please adjust your selection.")
    st.stop()

start_date = pd.to_datetime(start_date)
end_date   = pd.to_datetime(end_date)

# Filter datasets
mask = (store_data['date'] >= start_date) & (store_data['date'] <= end_date)
df_period = store_data.loc[mask].groupby('date')[['sales', 'onpromotion', 'transactions']].sum().reset_index()

# Previous period for growth calculation
period_length = end_date - start_date
prev_start = start_date - period_length
prev_mask = (store_data['date'] >= prev_start) & (store_data['date'] < start_date)
df_prev = store_data.loc[prev_mask].groupby('date')[['sales']].sum().reset_index()

# Calculate Metrics
total_sales = df_period['sales'].sum()
prev_sales = df_prev['sales'].sum()

if prev_sales > 0:
    growth_pct = ((total_sales - prev_sales) / prev_sales) * 100
else:
    growth_pct = 0.0

trend = "increasing" if growth_pct > 0 else "decreasing" if growth_pct < 0 else "stable"

avg_transactions = df_period['transactions'].mean() if not df_period.empty else 0
total_promotions = df_period['onpromotion'].sum()
promo_impact = "high" if total_promotions > df_period['sales'].mean() else "low"  # simple heuristic
seasonality = "yes"  # simplified

# Naive forecast
forecast_values = "Moderate Growth Expected" if trend == "increasing" else "Potential Decline"
forecast_growth = round(growth_pct * 0.5, 2)  # dummy projection
confidence = "medium"

# Top Row Metrics
c1, c2, c3, c4 = st.columns(4)
# 1. Define your logic for the Growth Card (c2)
growth_color = "#28a745" if growth_pct > 0 else "#dc3545"  # Green if positive, Red if negative

# 2. Define your logic for the Sales Trend Card (c4)
# Reddish if "decreasing", Greenish if "increasing"
trend_lower = trend.lower()
trend_color = "#dc3545" if "decreas" in trend_lower else "#28a745"

# 3. Render the cards with dynamic styles
with c1:
    st.markdown(f'<div class="metric-card"><div class="metric-title">Total Sales</div><div class="metric-value">${total_sales:,.0f}</div></div>', unsafe_allow_html=True)

with c2:
    st.markdown(f'''
        <div class="metric-card" style="border-left: 5px solid {growth_color}; background-color: {growth_color}22;">
            <div class="metric-title">Growth (vs Prev Period)</div>
            <div class="metric-value" style="color: {growth_color};">{growth_pct:+.1f}%</div>
        </div>
    ''', unsafe_allow_html=True)

with c3:
    st.markdown(f'<div class="metric-card"><div class="metric-title">Avg Daily Transactions</div><div class="metric-value">{avg_transactions:,.0f}</div></div>', unsafe_allow_html=True)

with c4:
    st.markdown(f'''
        <div class="metric-card" style="border-left: 5px solid {trend_color}; background-color: {trend_color}22;">
            <div class="metric-title">Sales Trend</div>
            <div class="metric-value" style="color: {trend_color};">{trend.title()}</div>
        </div>
    ''', unsafe_allow_html=True)

# Chart
st.subheader("Daily Sales Tragectory")
fig = px.line(df_period, x='date', y='sales', title=f"Sales for Store {selected_store} ({start_date.date()} to {end_date.date()})", line_shape='spline')
fig.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(211, 211, 211, 0.5)',griddash='dot')
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(211, 211, 211, 0.5)',griddash='dot')
st.plotly_chart(fig, use_container_width=True)

# Preview Postgres DB view
with st.expander("Explore Postgres Database `sales_forecasting.master_data` View"):
    st.dataframe(df_period.head(50))

# Generate Prompt
try:
    with open("Prompt_files/final_prompt.txt", "r", encoding="utf-8") as f:
        prompt_template = f.read()
except FileNotFoundError:
    st.error("final_prompt.txt not found. Please ensure it is in the project directory.")
    st.stop()

filled_prompt = prompt_template \
    .replace("{start_date}", str(start_date.date())) \
    .replace("{end_date}", str(end_date.date())) \
    .replace("{total_sales}", f"${total_sales:,.0f}") \
    .replace("{growth_percentage}", f"{growth_pct:.1f}") \
    .replace("{increasing/decreasing/stable}", trend) \
    .replace("{increase/decrease/stable}", "stable") \
    .replace("{value}", f"{avg_transactions:,.0f}") \
    .replace("{increase/decrease}", "increase" if total_promotions > 0 else "stable") \
    .replace("{high/medium/low}", promo_impact) \
    .replace("{yes/no}", seasonality) \
    .replace("{describe}", "Expected higher demand strictly related to weekends/holidays.") \
    .replace("{describe if relevant}", "Minimal direct impact from oil prices in this scenario") \
    .replace("{forecast_values}", forecast_values) \
    .replace("{forecast_growth}", str(forecast_growth)) \
    .replace("{date}: {spike/drop reason if known}", "Most recent peak: Weekend high traffic effect")

st.divider()

if st.button("🧠 Generate AI Strategy Report", type="primary", use_container_width=True):
    if not GROQ_API_KEY:
        st.error("GROQ_API_KEY not found in .env. Please add it to generate reports.")
    else:
        with st.spinner("AI is analyzing the data..."):
            try:
                client = Groq(api_key=GROQ_API_KEY)
                chat_completion = client.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": filled_prompt,
                        }
                    ],
                    model="llama-3.3-70b-versatile", 
                    temperature=0.3,
                    max_tokens=1500
                )
                
                report = chat_completion.choices[0].message.content
                st.success("Analysis Complete!")
                st.markdown(report)
                
            except Exception as e:
                st.error(f"Failed to generate report: {e}")

with st.expander("View Raw Dynamic Prompt Passed to AI"):
    st.text(filled_prompt)

# ── Data Flow Documentation (from Data_Flow_Task.ipynb) ─────────────────────
st.divider()
with st.expander("📖 Data Flow Documentation — Full Pipeline Explained", expanded=False):
    import json

    nb_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data_Flow_Task.ipynb")
    try:
        with open(nb_path, "r", encoding="utf-8") as nb_file:
            notebook = json.load(nb_file)

        # Extract and join all markdown cell sources
        doc_parts = []
        for cell in notebook.get("cells", []):
            if cell.get("cell_type") == "markdown":
                source = cell.get("source", [])
                # source is a list of strings — join them
                text = "".join(source).strip()
                if text:
                    doc_parts.append(text)

        full_doc = "\n\n".join(doc_parts)

        if full_doc:
            st.markdown(full_doc)
        else:
            st.info("No markdown content found in the notebook.")

    except FileNotFoundError:
        st.warning("⚠️ `Data_Flow_Task.ipynb` not found in the project directory.")
    except Exception as doc_err:
        st.error(f"Failed to load documentation: {doc_err}")
