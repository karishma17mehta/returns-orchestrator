import os
import uuid
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

# ----------------------------
# Paths / Config
# ----------------------------
RAW_DIR = "data/olist_raw"

ORDERS_CSV = os.path.join(RAW_DIR, "olist_orders_dataset.csv")
ORDER_ITEMS_CSV = os.path.join(RAW_DIR, "olist_order_items_dataset.csv")
PRODUCTS_CSV = os.path.join(RAW_DIR, "olist_products_dataset.csv")
CAT_TRANS_CSV = os.path.join(RAW_DIR, "product_category_name_translation.csv")
PAYMENTS_CSV = os.path.join(RAW_DIR, "olist_order_payments_dataset.csv")

SCHEMA_SQL = "data/schema.sql"

DB_URL = os.getenv("DATABASE_URL")
if not DB_URL:
    raise RuntimeError(
        "DATABASE_URL not set. Create a .env file (copy from .env.example) and set DATABASE_URL."
    )

engine = create_engine(DB_URL)

# Tune these if needed
RETURN_FRAC = 0.18
RETURN_WINDOW_DAYS = 30
RETURN_DAYS_AFTER_DELIVERY_MIN = 1
RETURN_DAYS_AFTER_DELIVERY_MAX = 45
CHUNKSIZE = 5000

# ----------------------------
# Helpers
# ----------------------------
def require_files(paths):
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Missing required files:\n" + "\n".join([f"- {m}" for m in missing])
        )

def run_schema():
    if not os.path.exists(SCHEMA_SQL):
        raise FileNotFoundError(f"Missing schema file: {SCHEMA_SQL}")
    with engine.begin() as conn:
        schema = open(SCHEMA_SQL, "r", encoding="utf-8").read()
        conn.execute(text(schema))

def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def to_sql_fast(df: pd.DataFrame, table: str):
    # Keep parameter count below Postgres limit (~65535)
    # params ≈ rows_in_chunk * num_columns
    ncols = df.shape[1]
    max_params = 60000  # stay safely under the limit
    safe_chunk = max(500, min(2000, max_params // max(ncols, 1)))

    print(f"   -> inserting {table} with chunksize={safe_chunk}, cols={ncols}, rows={len(df):,}")

    df.to_sql(
        table,
        engine,
        if_exists="append",
        index=False,
        chunksize=safe_chunk,
        method="multi",
    )


# ----------------------------
# Main
# ----------------------------
def main():
    print("🔎 Checking required files...")
    require_files([ORDERS_CSV, ORDER_ITEMS_CSV, PRODUCTS_CSV, CAT_TRANS_CSV, PAYMENTS_CSV, SCHEMA_SQL])

    print("1) Creating schema...")
    run_schema()

    print("2) Loading raw Olist files...")
    orders_raw = load_csv(ORDERS_CSV)
    items_raw = load_csv(ORDER_ITEMS_CSV)
    products_raw = load_csv(PRODUCTS_CSV)
    cat_raw = load_csv(CAT_TRANS_CSV)
    pay_raw = load_csv(PAYMENTS_CSV)

    # ----------------------------
    # PRODUCTS
    # ----------------------------
    # Olist products has 'product_category_name' (pt), translation maps pt->english
    products_joined = products_raw.merge(cat_raw, how="left", on="product_category_name")

    products_df = products_joined.rename(
        columns={"product_category_name_english": "category"}
    )[["product_id", "category"]].copy()

    # Price proxy: mean item price per product from order_items
    # (Olist order_items has column 'price')
    if "price" not in items_raw.columns:
        raise KeyError("Expected 'price' column in olist_order_items_dataset.csv")

    item_prices = items_raw.groupby("product_id")["price"].mean().reset_index()
    item_prices = item_prices.rename(columns={"price": "price"})

    products_df = products_df.merge(item_prices, how="left", on="product_id")
    products_df["category"] = products_df["category"].fillna("unknown")

    median_price = products_df["price"].median()
    if pd.isna(median_price):
        median_price = 0
    products_df["price"] = products_df["price"].fillna(median_price)

    # Deduplicate (safety)
    products_df = products_df.drop_duplicates(subset=["product_id"])

    # ----------------------------
    # ORDERS
    # ----------------------------
    required_order_cols = [
        "order_id",
        "customer_id",
        "order_status",
        "order_purchase_timestamp",
        "order_delivered_customer_date",
    ]
    for c in required_order_cols:
        if c not in orders_raw.columns:
            raise KeyError(f"Expected '{c}' in olist_orders_dataset.csv")

    orders_df = orders_raw[required_order_cols].copy()

    orders_df["order_date"] = pd.to_datetime(
        orders_df["order_purchase_timestamp"], errors="coerce"
    ).dt.date

    orders_df["delivered_date"] = pd.to_datetime(
        orders_df["order_delivered_customer_date"], errors="coerce"
    ).dt.date

    orders_df = orders_df.rename(columns={"order_status": "status"})
    orders_df = orders_df[["order_id", "customer_id", "order_date", "delivered_date", "status"]]

    # Deduplicate order_id to prevent PK insert failures
    orders_df = orders_df.drop_duplicates(subset=["order_id"])

    # Total amount from payments (sum per order)
    if "payment_value" not in pay_raw.columns:
        raise KeyError("Expected 'payment_value' in olist_order_payments_dataset.csv")

    pay_sum = pay_raw.groupby("order_id")["payment_value"].sum().reset_index()
    pay_sum = pay_sum.rename(columns={"payment_value": "total_amount"})

    orders_df = orders_df.merge(pay_sum, how="left", on="order_id")
    orders_df["total_amount"] = orders_df["total_amount"].fillna(0)

    # ----------------------------
    # ORDER ITEMS
    # ----------------------------
    required_item_cols = ["order_id", "order_item_id", "product_id", "price"]
    for c in required_item_cols:
        if c not in items_raw.columns:
            raise KeyError(f"Expected '{c}' in olist_order_items_dataset.csv")

    items_df = items_raw[required_item_cols].copy()
    items_df["quantity"] = 1
    items_df = items_df.rename(columns={"price": "item_price"})

    # Safety: drop duplicates on PK
    items_df = items_df.drop_duplicates(subset=["order_id", "order_item_id"])

    # Ensure order_items only reference orders we loaded (prevents orphan rows)
    items_df = items_df[items_df["order_id"].isin(orders_df["order_id"])].copy()

    # ----------------------------
    # SYNTHETIC RETURNS
    # ----------------------------
    delivered = orders_df[
        (orders_df["status"] == "delivered") & (orders_df["delivered_date"].notna())
    ].copy()

    np.random.seed(42)

    if len(delivered) == 0:
        returns_df = pd.DataFrame(columns=["return_id", "order_id", "return_date", "reason", "refund_amount", "is_late"])
    else:
        returned = delivered.sample(frac=RETURN_FRAC, random_state=42).copy()

        # return_date between 1 and 45 days after delivery
        days_after = np.random.randint(
            RETURN_DAYS_AFTER_DELIVERY_MIN,
            RETURN_DAYS_AFTER_DELIVERY_MAX + 1,
            size=len(returned),
        )

        returned["return_date"] = (
            pd.to_datetime(returned["delivered_date"])
            + pd.to_timedelta(days_after, unit="D")
        )
        returned["return_date"] = returned["return_date"].dt.date

        # reasons distribution
        returned["reason"] = np.random.choice(
            ["Defective", "Size/Fit", "Changed Mind", "Late Delivery", "Wrong Item"],
            size=len(returned),
            p=[0.22, 0.28, 0.25, 0.15, 0.10],
        )

        # refund_amount ~ total_amount * (0.6 to 1.0)
        multipliers = np.random.uniform(0.6, 1.0, size=len(returned))
        returned["refund_amount"] = (returned["total_amount"] * multipliers).round(2)

        # late if return_date > delivered_date + 30 days
        returned["is_late"] = (
            pd.to_datetime(returned["return_date"])
            > (pd.to_datetime(returned["delivered_date"]) + pd.to_timedelta(RETURN_WINDOW_DAYS, unit="D"))
        )

        returns_df = returned[["order_id", "return_date", "reason", "refund_amount", "is_late"]].copy()
        returns_df.insert(0, "return_id", [str(uuid.uuid4()) for _ in range(len(returns_df))])

    # ----------------------------
    # WRITE TO POSTGRES
    # ----------------------------
    print("3) Writing to Postgres (chunked inserts)...")
    to_sql_fast(products_df, "products")
    to_sql_fast(orders_df, "orders")

    print("   -> inserting order_items with chunksize=5000 (no multi)")
    items_df.to_sql("order_items", engine, if_exists="append", index=False, chunksize=5000)

    to_sql_fast(returns_df, "returns")

    # ----------------------------
    # VALIDATION
    # ----------------------------
    print("4) Basic validation...")
    with engine.begin() as conn:
        for tbl in ["products", "orders", "order_items", "returns"]:
            n = conn.execute(text(f"SELECT COUNT(*) FROM {tbl}")).scalar() or 0
            print(f"  - {tbl}: {n:,} rows")

        total_orders = conn.execute(text("SELECT COUNT(*) FROM orders")).scalar() or 0
        total_returns = conn.execute(text("SELECT COUNT(*) FROM returns")).scalar() or 0
        rr = (total_returns / max(total_orders, 1)) if total_orders else 0
        print(f"\nReturn rate (synthetic): {rr:.2%}")

        # Quick sanity: returns should reference valid orders
        bad = conn.execute(text("""
            SELECT COUNT(*) FROM returns r
            LEFT JOIN orders o ON r.order_id = o.order_id
            WHERE o.order_id IS NULL
        """)).scalar() or 0

        if bad > 0:
            print(f"⚠️ Warning: {bad} returns rows do not match orders (should be 0).")
        else:
            print("✅ Returns foreign key integrity looks good (0 unmatched).")

    print("\n✅ Day 1 complete: DB loaded + synthetic returns generated.")

if __name__ == "__main__":
    main()
