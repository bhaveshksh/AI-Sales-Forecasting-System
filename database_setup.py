import pandas as pd
from sqlalchemy import create_engine, text
import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()
DB_USERNAME = os.getenv("DB_USERNAME", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "admin")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "postgres")

# Connect via Psycopg2 for fast COPY
conn_str = f"dbname={DB_NAME} user={DB_USERNAME} password={DB_PASSWORD} host={DB_HOST} port={DB_PORT}"
conn = psycopg2.connect(conn_str)
conn.autocommit = True
cur = conn.cursor()

print("Creating schema 'sales_forecasting'...")
cur.execute("CREATE SCHEMA IF NOT EXISTS sales_forecasting;")

print("Cleaning up old tables to prevent conflicts...")
tables = ["master_data", "train", "oil", "stores", "transactions", "holidays"]
for t in tables:
    cur.execute(f"DROP TABLE IF EXISTS sales_forecasting.{t} CASCADE;")

# 1. Train Table Setup (Using BLAZING FAST COPY_EXPERT)
print("Creating and Loading train.csv (Fast COPY)...")
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
print(" - train table loaded successfully (in seconds!)")

# 2. Stores Setup
print("Creating and Loading stores.csv...")
cur.execute("""
    CREATE TABLE sales_forecasting.stores (
        store_nbr INT PRIMARY KEY,
        city TEXT,
        state TEXT,
        type TEXT,
        cluster INT
    );
""")
with open("Datasets/stores.csv", "r") as f:
    cur.copy_expert("COPY sales_forecasting.stores FROM STDIN WITH CSV HEADER", f)

# 3. Transactions Setup
print("Creating and Loading transactions.csv...")
cur.execute("""
    CREATE TABLE sales_forecasting.transactions (
        date DATE,
        store_nbr INT,
        transactions INT
    );
""")
with open("Datasets/transactions.csv", "r") as f:
    cur.copy_expert("COPY sales_forecasting.transactions FROM STDIN WITH CSV HEADER", f)

# 4. Holidays Setup
print("Creating and Loading holidays_events.csv...")
cur.execute("""
    CREATE TABLE sales_forecasting.holidays (
        date DATE,
        type TEXT,
        locale TEXT,
        locale_name TEXT,
        description TEXT,
        transferred BOOLEAN
    );
""")
with open("Datasets/holidays_events.csv", "r", encoding="utf-8") as f:
    cur.copy_expert("COPY sales_forecasting.holidays FROM STDIN WITH CSV HEADER", f)

# 5. Oil Setup (ffill with Pandas, load with COPY for proper DATE type)
print("Processing and Loading oil.csv (ffill + COPY)...")
df_oil = pd.read_csv("Datasets/oil.csv")
df_oil['dcoilwtico'] = df_oil['dcoilwtico'].ffill()
cur.execute("""
    CREATE TABLE IF NOT EXISTS sales_forecasting.oil (
        date DATE PRIMARY KEY,
        dcoilwtico FLOAT
    );
""")
import io
buf = io.StringIO()
df_oil.to_csv(buf, index=False)
buf.seek(0)
cur.copy_expert("COPY sales_forecasting.oil FROM STDIN WITH CSV HEADER", buf)

print("Creating master_data table via JOIN...")
create_table_query = """
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
LEFT JOIN sales_forecasting.stores s
ON t.store_nbr = s.store_nbr
LEFT JOIN sales_forecasting.transactions tr
ON t.date = tr.date::DATE AND t.store_nbr = tr.store_nbr
LEFT JOIN sales_forecasting.oil o
ON t.date = o.date::DATE
LEFT JOIN sales_forecasting.holidays h
ON t.date = h.date::DATE;
"""
cur.execute(create_table_query)

cur.close()
conn.close()

print("✅ ETL Pipeline Completed! Everything loaded instantly. Refresh Streamlit!")
