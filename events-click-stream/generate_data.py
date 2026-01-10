import random
import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta

INACTIVITY_MINUTES = 30

# Constants
ALLOWED_DEVICES = ["Web", "Android", "iOS"]
ALLOWED_CITIES = ["Delhi", "Mumbai", "Bangalore", "Pune", "Chennai"]
ALLOWED_EVENTS = ["landing", "product_view", "add_to_cart", "checkout", "payment", "logout"]


def make_user_ids(n=20):
    return [f"U{str(i).zfill(3)}" for i in range(1, n+1)]


def make_product_ids(n=20):
    return [f"P{str(i).zfill(3)}" for i in range(1, n+1)]


USERS = make_user_ids(20)
PRODUCTS = make_product_ids(20)


def fmt_amount(x: float):
    if random.random() < 0.6:
        return f"₹{int(round(x)):,}"
    return str(round(x, 2))

def gen_valid_funnel_session(user_id: str, start: datetime):
    events = []
    t = start
    def add(ev, dt=2, product=None, amount=None):
        nonlocal t
        t = t + timedelta(minutes=dt)
        events.append({
            "user_id": user_id,
            "event_time": t.isoformat(timespec="seconds"),
            "event_name": ev,
            "device": random.choice(ALLOWED_DEVICES),
            "city": random.choice(ALLOWED_CITIES),
            "session_hint": "",
            "product_id": product if product else "",
            "amount": amount if amount else "",
        })
    add("landing", dt=0)
    add("product_view", dt=3, product=random.choice(PRODUCTS))
    add("add_to_cart", dt=2, product=random.choice(PRODUCTS))
    add("checkout", dt=4)
    add("payment", dt=2, amount=fmt_amount(random.uniform(199, 2999)))
    add("logout", dt=1)
    return events


def gen_random_events(user_id: str, start: datetime, count: int):
    events = []
    t = start
    for _ in range(count):
        t = t + timedelta(minutes=random.randint(1, 18))
        ev = random.choice(ALLOWED_EVENTS)
        product = random.choice(PRODUCTS) if ev in ["product_view","add_to_cart"] and random.random() > 0.15 else ""
        amount = fmt_amount(random.uniform(99, 4999)) if ev == "payment" and random.random() > 0.15 else ""
        events.append({
            "user_id": user_id,
            "event_time": t.isoformat(timespec="seconds"),
            "event_name": ev,
            "device": random.choice(ALLOWED_DEVICES),
            "city": random.choice(ALLOWED_CITIES),
            "session_hint": "" if random.random() < 0.8 else "maybe_session",
            "product_id": product,
            "amount": amount,
        })
    return events


base = datetime(2025, 12, 20, 9, 0, 0)


rows = []


# Ensure at least 5 users complete the funnel properly
funnel_users = random.sample(USERS, 5)
for u in funnel_users:
    rows.extend(gen_valid_funnel_session(u, base + timedelta(hours=random.randint(0, 48))))


# Generate more random events for all users
for u in USERS:
    rows.extend(gen_random_events(u, base + timedelta(hours=random.randint(0, 72)), count=random.randint(2, 6)))


df_raw = pd.DataFrame(rows)


# Add impossible sequences
df_raw = pd.concat([df_raw, pd.DataFrame([{
    "user_id": USERS[0], "event_time": (base + timedelta(hours=1)).isoformat(timespec="seconds"),
    "event_name": "payment", "device": "Web", "city": "Delhi", "session_hint": "", "product_id": "", "amount": "₹999"
}])], ignore_index=True)


df_raw = pd.concat([df_raw, pd.DataFrame([{
    "user_id": USERS[1], "event_time": (base + timedelta(hours=2)).isoformat(timespec="seconds"),
    "event_name": "add_to_cart", "device": "Android", "city": "Mumbai", "session_hint": "", "product_id": "P005", "amount": ""
}])], ignore_index=True)


df_raw = pd.concat([df_raw, pd.DataFrame([{
    "user_id": USERS[2], "event_time": (base + timedelta(hours=3)).isoformat(timespec="seconds"),
    "event_name": "checkout", "device": "iOS", "city": "Pune", "session_hint": "", "product_id": "", "amount": ""
}])], ignore_index=True)


# Malformed timestamps
bad_times = ["not_a_time", "2025-13-01T10:00:00", "2025-12-32T09:00:00"]
for bt in bad_times:
    df_raw.loc[random.randint(0, len(df_raw)-1), "event_time"] = bt


# Missing product_id for product events
for _ in range(5):
    idx = random.randint(0, len(df_raw)-1)
    if df_raw.loc[idx, "event_name"] in ["product_view","add_to_cart"]:
        df_raw.loc[idx, "product_id"] = ""


# Missing amount for payments
for _ in range(5):
    idx = random.randint(0, len(df_raw)-1)
    if df_raw.loc[idx, "event_name"] == "payment":
        df_raw.loc[idx, "amount"] = ""


# Duplicates (>=5)
dup_indices = random.sample(range(len(df_raw)), 5)
df_dups = df_raw.iloc[dup_indices].copy()
df_raw = pd.concat([df_raw, df_dups], ignore_index=True)


# Shuffle for out-of-order
df_raw = df_raw.sample(frac=1.0, random_state=42).reset_index(drop=True)


assert len(df_raw) >= 80, f"Dataset too small: {len(df_raw)} rows"
df_raw.to_csv("events_raw.csv", index=False)


print("✅ Wrote events_raw.csv")
print("Rows:", len(df_raw), "| Users:", df_raw['user_id'].nunique())
df_raw.head(10)


data_contract = {
    "user_id": {"type": "string", "pattern": r"^U\d{3}$", "required": True},
    "event_time": {"type": "datetime", "required": True, "note": "Parseable ISO timestamp; invalid -> NaT"},
    "event_name": {"type": "enum", "allowed": ALLOWED_EVENTS, "required": True},
    "device": {"type": "enum", "allowed": ALLOWED_DEVICES, "required": True},
    "city": {"type": "enum", "allowed": ALLOWED_CITIES, "required": True},
    "session_hint": {"type": "string", "required": False},
    "product_id": {"type": "string", "pattern": r"^P\d{3}$", "required_when": "event_name in {product_view, add_to_cart}"},
    "amount": {"type": "currency_string", "required_when": "event_name == payment"},
}


rules = [
    "R1: event_name must be one of allowed values",
    "R2: event_time must parse to a timestamp (invalid -> NaT but must be reported)",
    "R3: product_view and add_to_cart must have non-empty product_id (P###)",
    "R4: payment must have a non-empty amount (currency strings allowed)",
    "R5: sessionization uses user_id + sorted event_time; new session if gap > 30 minutes",
    "R6: funnel must be strictly ordered within a session: landing → product_view → add_to_cart → checkout → payment",
]


print("=== Data Contract ===")
for k,v in data_contract.items():
    print(k, "=>", v)
print("\n=== Rules ===")
for r in rules:
    print("-", r)


df = pd.read_csv("events_raw.csv")


# Normalize text fields
for col in ["user_id","event_name","device","city","session_hint","product_id","amount"]:
    df[col] = df[col].astype("string").fillna("").str.strip()


df["event_name"] = df["event_name"].str.lower()
df["device"] = df["device"].str.replace(r"\s+", " ", regex=True).str.title()
df["city"] = df["city"].str.replace(r"\s+", " ", regex=True).str.title()


# Parse timestamps safely
df["event_time_parsed"] = pd.to_datetime(df["event_time"], errors="coerce")


# Deduplicate
before = len(df)
df = df.drop_duplicates()
after = len(df)


def parse_amount_to_float(s):
    s = (str(s) if s is not None else "").strip()
    if s == "":
        return np.nan
    cleaned = re.sub(r"[^0-9.]", "", s)
    try:
        return float(cleaned) if cleaned != "" else np.nan
    except:
        return np.nan


df["amount_value"] = df["amount"].apply(parse_amount_to_float)


# Rule flags
df["flag_event_allowed"] = df["event_name"].isin(ALLOWED_EVENTS)
df["flag_timestamp_valid"] = df["event_time_parsed"].notna()
df["flag_product_required_ok"] = np.where(df["event_name"].isin(["product_view","add_to_cart"]),
                                         df["product_id"].str.match(r"^P\d{3}$"),
                                         True)
df["flag_payment_amount_ok"] = np.where(df["event_name"].eq("payment"),
                                       df["amount_value"].notna(),
                                       True)


df["flag_row_valid_core"] = df["flag_event_allowed"] & df["flag_product_required_ok"] & df["flag_payment_amount_ok"]


df_clean = df.copy()
df_clean.to_csv("events_clean.csv", index=False)


print("✅ Wrote events_clean.csv")
print("Rows before dedup:", before, "| after dedup:", after, "| removed:", before-after)
df_clean.head(10)

df_sess = df_clean[df_clean["flag_timestamp_valid"]].copy()
df_sess = df_sess.sort_values(["user_id","event_time_parsed"]).reset_index(drop=True)


df_sess["prev_time"] = df_sess.groupby("user_id")["event_time_parsed"].shift(1)
df_sess["gap_min"] = (df_sess["event_time_parsed"] - df_sess["prev_time"]).dt.total_seconds() / 60.0
df_sess["new_session"] = (df_sess["gap_min"].isna()) | (df_sess["gap_min"] > INACTIVITY_MINUTES)


df_sess["session_num"] = df_sess.groupby("user_id")["new_session"].cumsum()
df_sess["session_id"] = df_sess["user_id"] + "_S" + df_sess["session_num"].astype(int).astype(str).str.zfill(2)


session_bounds = df_sess.groupby("session_id").agg(
    user_id=("user_id","first"),
    session_start=("event_time_parsed","min"),
    session_end=("event_time_parsed","max"),
    events=("event_name","count")
).reset_index()


session_bounds.to_csv("sessions.csv", index=False)
print("✅ Wrote sessions.csv | Sessions:", len(session_bounds))
session_bounds.head(10)

FUNNEL = ["landing","product_view","add_to_cart","checkout","payment"]


step_times = (df_sess[df_sess["event_name"].isin(FUNNEL)]
              .groupby(["session_id","event_name"])["event_time_parsed"]
              .min()
              .unstack("event_name"))


def ordered_ok(row, steps):
    times = [row.get(s) for s in steps]
    if any(pd.isna(t) for t in times):
        return False
    return all(times[i] <= times[i+1] for i in range(len(times)-1))


for step in FUNNEL:
    if step not in step_times.columns:
        step_times[step] = pd.NaT


step_times = step_times.reset_index()
step_times["reach_landing"] = step_times["landing"].notna()
step_times["reach_product_view"] = step_times.apply(lambda r: ordered_ok(r, ["landing","product_view"]), axis=1)
step_times["reach_add_to_cart"] = step_times.apply(lambda r: ordered_ok(r, ["landing","product_view","add_to_cart"]), axis=1)
step_times["reach_checkout"] = step_times.apply(lambda r: ordered_ok(r, ["landing","product_view","add_to_cart","checkout"]), axis=1)
step_times["reach_payment"] = step_times.apply(lambda r: ordered_ok(r, FUNNEL), axis=1)


metrics = {
    "sessions_total": int(step_times["session_id"].nunique()),
    "landing_sessions": int(step_times["reach_landing"].sum()),
    "product_view_sessions": int(step_times["reach_product_view"].sum()),
    "add_to_cart_sessions": int(step_times["reach_add_to_cart"].sum()),
    "checkout_sessions": int(step_times["reach_checkout"].sum()),
    "payment_sessions": int(step_times["reach_payment"].sum()),
}
funnel_df = pd.DataFrame([metrics])
funnel_df["pv_rate"] = funnel_df["product_view_sessions"] / funnel_df["landing_sessions"].replace({0: np.nan})
funnel_df["atc_rate"] = funnel_df["add_to_cart_sessions"] / funnel_df["product_view_sessions"].replace({0: np.nan})
funnel_df["checkout_rate"] = funnel_df["checkout_sessions"] / funnel_df["add_to_cart_sessions"].replace({0: np.nan})
funnel_df["payment_rate"] = funnel_df["payment_sessions"] / funnel_df["checkout_sessions"].replace({0: np.nan})


user_summary = (step_times.merge(session_bounds[["session_id","user_id"]], on="session_id", how="left")
                .groupby("user_id")[["reach_landing","reach_product_view","reach_add_to_cart","reach_checkout","reach_payment"]]
                .sum()
                .reset_index())


funnel_df.to_csv("funnel_metrics.csv", index=False)
user_summary.to_csv("user_funnel_summary.csv", index=False)


print("✅ Wrote funnel_metrics.csv and user_funnel_summary.csv")
print("\n=== Funnel Metrics ===")
print(funnel_df)
print("\n=== User Funnel Summary (first 10) ===")
print(user_summary.head(10))



dfv = df_clean.copy()


invalid_ts = int((~dfv["flag_timestamp_valid"]).sum())
invalid_event = int((~dfv["flag_event_allowed"]).sum())
missing_product = int((~dfv["flag_product_required_ok"]).sum())
missing_payment_amount = int((~dfv["flag_payment_amount_ok"]).sum())


seq_issues = []
for sid, g in df_sess.groupby("session_id"):
    g = g.sort_values("event_time_parsed")
    events = list(g["event_name"])
    def first_idx(ev):
        return events.index(ev) if ev in events else None
    idx_payment = first_idx("payment")
    idx_checkout = first_idx("checkout")
    idx_atc = first_idx("add_to_cart")
    idx_pv = first_idx("product_view")
    if idx_payment is not None and idx_checkout is None:
        seq_issues.append(("payment_without_checkout", sid))
    if idx_atc is not None and idx_pv is None:
        seq_issues.append(("add_to_cart_without_product_view", sid))
    if idx_checkout is not None and idx_atc is None:
        seq_issues.append(("checkout_without_add_to_cart", sid))
    if idx_atc is not None and idx_pv is not None and idx_atc < idx_pv:
        seq_issues.append(("add_to_cart_before_product_view", sid))
    if idx_payment is not None and idx_checkout is not None and idx_payment < idx_checkout:
        seq_issues.append(("payment_before_checkout", sid))


report = {
    "rows_after_dedup": len(dfv),
    "invalid_timestamps": invalid_ts,
    "invalid_event_names": invalid_event,
    "missing_product_id_where_required": missing_product,
    "missing_payment_amount_where_required": missing_payment_amount,
    "session_count": int(session_bounds["session_id"].nunique()),
    "sequence_issue_instances": len(seq_issues),
}
