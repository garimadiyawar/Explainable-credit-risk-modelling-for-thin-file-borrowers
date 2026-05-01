"""
generate_synthetic_data.py  (v2 — calibrated to match real Home Credit statistics)
-----------------------------------------------------------------------------------
Generates synthetic CSVs that closely match the REAL Home Credit dataset:
  - 30,000 applications (enough for gradient boosting to work properly)
  - 8.1% default rate  (matches real dataset)
  - ~13.5% thin-file borrowers (matches real dataset)
  - EXT_SOURCE features as primary drivers (matches literature)
  - Realistic feature correlations and distributions
"""

import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_DIR, RANDOM_SEED

rng = np.random.default_rng(RANDOM_SEED)
os.makedirs(DATA_DIR, exist_ok=True)

N_APP   = 30_000   # large enough for gradient boosting
TARGET_RATE = 0.081  # matches real dataset

# ── helpers ─────────────────────────────────────────────────────────────────
def clip(arr, lo, hi): return np.clip(arr, lo, hi)

# ── 1. application_train.csv ────────────────────────────────────────────────
def make_application():
    print(f"Generating application_train.csv  (n={N_APP:,}, target_rate≈{TARGET_RATE:.1%}) ...")
    N = N_APP

    # ── Core credit-relevant signals (determine default) ──────────────────
    age_years        = rng.uniform(21, 68, N)
    employed_years   = clip(rng.exponential(4, N), 0, 40)
    income           = rng.lognormal(11.2, 0.55, N)         # ~73k mean
    credit_amount    = income * rng.uniform(1.5, 9.0, N)
    annuity          = credit_amount / rng.uniform(12, 72, N)

    # EXT_SOURCE scores — primary predictors in real dataset
    ext2 = clip(rng.beta(4, 4, N), 0.01, 0.99)
    ext3 = clip(rng.beta(3, 5, N) + rng.normal(0, 0.05, N), 0.01, 0.99)
    ext1 = clip(rng.beta(3, 4, N) + rng.normal(0, 0.07, N), 0.01, 0.99)

    annuity_income   = annuity / (income + 1)
    credit_income    = credit_amount / (income + 1)
    employed_ratio   = employed_years / (age_years + 1)

    # ── Logistic model that produces ~8.1% default rate ───────────────────
    # Calibrated intercept so mean(sigmoid(logit)) ≈ 0.081
    logit = (
        -4.20
        + 2.80 * (1 - ext2)
        + 2.00 * (1 - ext3)
        + 1.20 * (1 - ext1)
        + 1.50 * annuity_income
        + 0.80 * np.log1p(credit_income)
        - 1.20 * employed_ratio
        - 0.60 * (age_years / 65)
        + rng.normal(0, 0.8, N)          # individual noise
    )
    prob   = 1 / (1 + np.exp(-logit))
    # Force exactly 8.1% default rate via threshold calibration
    threshold = np.percentile(prob, 100 * (1 - TARGET_RATE))
    target    = (prob >= threshold).astype(int)

    # ── Categoricals ──────────────────────────────────────────────────────
    edu_choices    = ["Secondary / secondary special","Higher education",
                      "Incomplete higher","Lower secondary","Academic degree"]
    edu_weights    = [0.710, 0.220, 0.050, 0.015, 0.005]
    family_choices = ["Married","Single / not married","Civil marriage","Separated","Widow"]
    family_weights = [0.630, 0.280, 0.060, 0.020, 0.010]
    income_types   = ["Working","Commercial associate","Pensioner","State servant","Unemployed"]
    income_weights = [0.510, 0.230, 0.180, 0.075, 0.005]

    # sentinel: 18.5% of employed are pensioners → DAYS_EMPLOYED = 365243
    days_emp_raw = -employed_years * 365
    sentinel_mask = rng.random(N) < 0.185
    days_emp_out  = np.where(sentinel_mask, 365243.0, days_emp_raw)

    df = pd.DataFrame({
        "SK_ID_CURR": np.arange(100_000, 100_000 + N),
        "TARGET":     target,

        # Contract / demographics
        "NAME_CONTRACT_TYPE": rng.choice(["Cash loans","Revolving loans"], N, p=[0.90,0.10]),
        "CODE_GENDER":        rng.choice(["M","F","XNA"], N, p=[0.360,0.635,0.005]),
        "FLAG_OWN_CAR":       rng.choice(["Y","N"], N, p=[0.34,0.66]),
        "FLAG_OWN_REALTY":    rng.choice(["Y","N"], N, p=[0.69,0.31]),
        "CNT_CHILDREN":       rng.integers(0, 5, N),
        "CNT_FAM_MEMBERS":    (rng.integers(1, 6, N)).astype(float),

        # Financial
        "AMT_INCOME_TOTAL":   income,
        "AMT_CREDIT":         credit_amount,
        "AMT_ANNUITY":        annuity,
        "AMT_GOODS_PRICE":    credit_amount * rng.uniform(0.80, 1.00, N),

        # Employment / age
        "DAYS_BIRTH":         (-age_years * 365).astype(int),
        "DAYS_EMPLOYED":      days_emp_out,
        "DAYS_REGISTRATION":  -rng.integers(0, 25000, N).astype(float),
        "DAYS_ID_PUBLISH":    -rng.integers(0, 7000,  N).astype(float),

        # Categorical
        "NAME_INCOME_TYPE":      rng.choice(income_types, N, p=income_weights),
        "NAME_EDUCATION_TYPE":   rng.choice(edu_choices,  N, p=edu_weights),
        "NAME_FAMILY_STATUS":    rng.choice(family_choices, N, p=family_weights),
        "NAME_HOUSING_TYPE":     rng.choice(["House / apartment","With parents",
                                              "Municipal apartment","Rented apartment"],
                                             N, p=[0.89,0.05,0.04,0.02]),
        "NAME_TYPE_SUITE":       rng.choice(["Unaccompanied","Family","Spouse, partner"],
                                             N, p=[0.55,0.40,0.05]),
        "ORGANIZATION_TYPE":     rng.choice(["Business Entity Type 3","School",
                                              "Government","Transport: type 4","Other"],
                                             N, p=[0.23,0.15,0.12,0.08,0.42]),
        "OCCUPATION_TYPE":       pd.array(np.where(rng.random(N) < 0.31, None,
                                           rng.choice(["Laborers","Core staff","Accountants",
                                                        "Managers","Drivers","Sales staff"],
                                                        N, p=[0.33,0.20,0.12,0.10,0.09,0.16])),
                                           dtype=object),
        "WEEKDAY_APPR_PROCESS_START": rng.choice(["MONDAY","TUESDAY","WEDNESDAY",
                                                    "THURSDAY","FRIDAY","SATURDAY","SUNDAY"], N),
        "HOUR_APPR_PROCESS_START":    rng.integers(0, 24, N),
        "FONDKAPREMONT_MODE":  rng.choice(["reg oper spec account","reg oper account",
                                            "org spec account", None], N, p=[0.30,0.30,0.20,0.20]),
        "HOUSETYPE_MODE":      rng.choice(["block of flats","terraced house",
                                            "specific housing", None], N, p=[0.85,0.04,0.04,0.07]),
        "WALLSMATERIAL_MODE":  rng.choice(["Panel","Stone, brick","Block","Wooden",np.nan],
                                           N, p=[0.30,0.40,0.10,0.05,0.15]),
        "EMERGENCYSTATE_MODE": rng.choice(["No","Yes",None], N, p=[0.85,0.01,0.14]),

        # EXT_SOURCE (partially missing — matches real dataset)
        "EXT_SOURCE_1": np.where(rng.random(N) < 0.56, np.nan, ext1),
        "EXT_SOURCE_2": np.where(rng.random(N) < 0.04, np.nan, ext2),  # 4% missing
        "EXT_SOURCE_3": np.where(rng.random(N) < 0.20, np.nan, ext3),

        # Flags
        "FLAG_MOBIL":        np.ones(N, dtype=int),
        "FLAG_EMP_PHONE":    rng.integers(0, 2, N),
        "FLAG_WORK_PHONE":   rng.integers(0, 2, N),
        "FLAG_CONT_MOBILE":  rng.integers(0, 2, N),
        "FLAG_PHONE":        rng.integers(0, 2, N),
        "FLAG_EMAIL":        rng.integers(0, 2, N),
        "OWN_CAR_AGE":       np.where(rng.random(N) < 0.66, np.nan, rng.uniform(0,30,N)),

        # Address mismatches
        "REG_REGION_NOT_LIVE_REGION":  rng.integers(0, 2, N),
        "REG_REGION_NOT_WORK_REGION":  rng.integers(0, 2, N),
        "LIVE_REGION_NOT_WORK_REGION": rng.integers(0, 2, N),
        "REG_CITY_NOT_LIVE_CITY":      rng.integers(0, 2, N),
        "REG_CITY_NOT_WORK_CITY":      rng.integers(0, 2, N),
        "LIVE_CITY_NOT_WORK_CITY":     rng.integers(0, 2, N),

        # Document flags
        "FLAG_DOCUMENT_2": rng.integers(0, 2, N),
        "FLAG_DOCUMENT_3": rng.integers(0, 2, N),
        "FLAG_DOCUMENT_4": rng.integers(0, 2, N),
        "FLAG_DOCUMENT_5": rng.integers(0, 2, N),
        "FLAG_DOCUMENT_6": rng.integers(0, 2, N),

        # Bureau enquiry counts (partially missing)
        "AMT_REQ_CREDIT_BUREAU_HOUR": np.where(rng.random(N)<0.13, np.nan, rng.integers(0,4,N).astype(float)),
        "AMT_REQ_CREDIT_BUREAU_DAY":  np.where(rng.random(N)<0.13, np.nan, rng.integers(0,3,N).astype(float)),
        "AMT_REQ_CREDIT_BUREAU_WEEK": np.where(rng.random(N)<0.13, np.nan, rng.integers(0,5,N).astype(float)),
        "AMT_REQ_CREDIT_BUREAU_MON":  np.where(rng.random(N)<0.13, np.nan, rng.integers(0,9,N).astype(float)),
        "AMT_REQ_CREDIT_BUREAU_QRT":  np.where(rng.random(N)<0.13, np.nan, rng.integers(0,8,N).astype(float)),
        "AMT_REQ_CREDIT_BUREAU_YEAR": np.where(rng.random(N)<0.13, np.nan, rng.integers(0,25,N).astype(float)),

        # Apartment features (48% missing like real data)
        "APARTMENTS_AVG":                np.where(rng.random(N)<0.48, np.nan, rng.uniform(0,1,N)),
        "BASEMENTAREA_AVG":              np.where(rng.random(N)<0.49, np.nan, rng.uniform(0,1,N)),
        "YEARS_BEGINEXPLUATATION_AVG":   np.where(rng.random(N)<0.48, np.nan, rng.uniform(0.9,1,N)),
        "REGION_POPULATION_RELATIVE":    rng.uniform(0.0001, 0.072, N),
        "REGION_RATING_CLIENT":          rng.integers(1, 4, N),
        "REGION_RATING_CLIENT_W_CITY":   rng.integers(1, 4, N),
        "DAYS_LAST_PHONE_CHANGE":       -rng.integers(0, 4000, N).astype(float),
    })

    df.to_csv(os.path.join(DATA_DIR, "application_train.csv"), index=False)
    actual_rate = df["TARGET"].mean()
    thin_proxy  = df["EXT_SOURCE_2"].isna().mean()  # rough proxy
    print(f"  Saved. shape={df.shape}  default_rate={actual_rate:.4f}  (target {TARGET_RATE:.3f})")
    return df["SK_ID_CURR"].values, df[["SK_ID_CURR","TARGET","EXT_SOURCE_2"]].copy()


# ── 2. bureau.csv  (86.5% of applicants have records → 13.5% thin-file) ────
def make_bureau(app_ids, app_df):
    print("Generating bureau.csv ...")
    N = len(app_ids)
    # 86.5% have bureau records — exactly matches real dataset
    has_bureau = rng.choice(app_ids, size=int(0.865 * N), replace=False)

    rows = []
    bid  = 1_000_000
    for sid in has_bureau:
        n_loans = rng.integers(1, 7)
        for _ in range(n_loans):
            is_active = rng.random() < 0.40
            rows.append({
                "SK_ID_CURR":          sid,
                "SK_ID_BUREAU":        bid,
                "CREDIT_ACTIVE":       "Active" if is_active else rng.choice(["Closed","Sold"], p=[0.92,0.08]),
                "CREDIT_CURRENCY":     "currency 1",
                "DAYS_CREDIT":        -int(rng.integers(0, 2000)),
                "CREDIT_DAY_OVERDUE":  int(rng.integers(0, 90)) if rng.random() < 0.12 else 0,
                "DAYS_CREDIT_ENDDATE": int(rng.integers(-500, 2000)),
                "DAYS_ENDDATE_FACT":   float(-rng.integers(0, 2000)) if not is_active else np.nan,
                "AMT_CREDIT_MAX_OVERDUE": float(rng.lognormal(8, 1)) if rng.random() < 0.15 else np.nan,
                "CNT_CREDIT_PROLONG":  int(rng.integers(0, 3)),
                "AMT_CREDIT_SUM":      float(rng.lognormal(11.0, 0.8)),
                "AMT_CREDIT_SUM_DEBT": float(rng.lognormal(10.0, 0.9)) if rng.random() < 0.45 else 0.0,
                "AMT_CREDIT_SUM_LIMIT":float(rng.lognormal(11.2, 0.7)) if rng.random() < 0.35 else np.nan,
                "AMT_CREDIT_SUM_OVERDUE": 0.0,
                "CREDIT_TYPE":         rng.choice(["Consumer credit","Credit card",
                                                    "Mortgage","Car loan"], p=[0.50,0.20,0.20,0.10]),
                "DAYS_CREDIT_UPDATE": -int(rng.integers(0, 500)),
                "AMT_ANNUITY":         float(rng.lognormal(9.0, 0.8)) if rng.random() < 0.55 else np.nan,
            })
            bid += 1

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(DATA_DIR, "bureau.csv"), index=False)
    n_thin = N - len(has_bureau)
    print(f"  Saved. shape={df.shape}  thin-file applicants={n_thin:,} ({n_thin/N:.1%})")
    return df["SK_ID_BUREAU"].values


# ── 3. bureau_balance.csv ───────────────────────────────────────────────────
def make_bureau_balance(bureau_ids):
    print("Generating bureau_balance.csv ...")
    sample_bids = rng.choice(bureau_ids, size=min(6000, len(bureau_ids)), replace=False)
    rows = []
    for bid in sample_bids:
        n_months = rng.integers(3, 36)
        for m in range(-n_months, 0):
            status = rng.choice(["C","0","1","2","3","X"], p=[0.20,0.58,0.10,0.05,0.02,0.05])
            rows.append({"SK_ID_BUREAU": bid, "MONTHS_BALANCE": int(m), "STATUS": status})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(DATA_DIR, "bureau_balance.csv"), index=False)
    print(f"  Saved. shape={df.shape}")


# ── 4. previous_application.csv ─────────────────────────────────────────────
def make_previous(app_ids):
    print("Generating previous_application.csv ...")
    # Each of 80% of applicants has 1-4 previous applications
    sample = rng.choice(app_ids, size=int(0.80 * len(app_ids)), replace=False)
    rows = []
    pid  = 2_000_000
    for sid in sample:
        n_prev = rng.integers(1, 5)
        for _ in range(n_prev):
            credit = float(rng.lognormal(11.0, 0.8))
            status = rng.choice(["Approved","Refused","Canceled","Unused offer"],
                                  p=[0.62, 0.18, 0.12, 0.08])
            rows.append({
                "SK_ID_PREV":                  pid,
                "SK_ID_CURR":                  sid,
                "NAME_CONTRACT_TYPE":          rng.choice(["Consumer loans","Cash loans","Revolving loans"], p=[0.5,0.4,0.1]),
                "AMT_ANNUITY":                 credit / float(rng.integers(12, 72)),
                "AMT_APPLICATION":             credit * float(rng.uniform(0.90, 1.10)),
                "AMT_CREDIT":                  credit,
                "AMT_DOWN_PAYMENT":            float(rng.uniform(0, credit*0.2)) if rng.random()<0.3 else np.nan,
                "AMT_GOODS_PRICE":             credit * float(rng.uniform(0.8, 1.0)),
                "NAME_PAYMENT_TYPE":           rng.choice(["Cash through the bank","Non-cash from your account"], p=[0.7,0.3]),
                "DAYS_DECISION":              -int(rng.integers(0, 2000)),
                "NAME_CONTRACT_STATUS":        status,
                "NAME_CLIENT_TYPE":            rng.choice(["Repeater","New","Refreshed"], p=[0.70,0.25,0.05]),
                "NAME_GOODS_CATEGORY":         rng.choice(["Mobile","Consumer Electronics","Furniture","Other"], p=[0.30,0.25,0.20,0.25]),
                "CHANNEL_TYPE":                rng.choice(["Credit and cash offices","Country-wide","Stone"], p=[0.5,0.3,0.2]),
                "HOUR_APPR_PROCESS_START":     int(rng.integers(0, 24)),
                "CNT_PAYMENT":                 int(rng.integers(0, 36)),
                "NAME_YIELD_GROUP":            rng.choice(["XNA","low_action","low_normal","middle","high"], p=[0.2]*5),
                "PRODUCT_COMBINATION":         "POS mobile with interest",
                "DAYS_FIRST_DRAWING":          np.nan,
                "DAYS_FIRST_DUE":             -int(rng.integers(0, 730)),
                "DAYS_LAST_DUE_1ST_VERSION":  -int(rng.integers(0, 1500)),
                "DAYS_LAST_DUE":              -int(rng.integers(0, 1500)),
                "DAYS_TERMINATION":           -int(rng.integers(0, 1500)),
                "NFLAG_INSURED_ON_APPROVAL":   int(rng.integers(0, 2)),
                "RATE_INTEREST_PRIMARY":       float(rng.uniform(0.05, 0.40)) if rng.random()<0.5 else np.nan,
                "RATE_INTEREST_PRIVILEGED":    float(rng.uniform(0.01, 0.20)) if rng.random()<0.3 else np.nan,
                "NAME_TYPE_SUITE":             rng.choice(["Unaccompanied","Family"], p=[0.6,0.4]),
                "NAME_SELLER_INDUSTRY":        rng.choice(["Connectivity","Consumer electronics","Clothing"], p=[0.4,0.4,0.2]),
            })
            pid += 1
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(DATA_DIR, "previous_application.csv"), index=False)
    print(f"  Saved. shape={df.shape}")
    return df["SK_ID_PREV"].values


# ── 5. installments_payments.csv  (key alternative feature source) ──────────
def make_installments(app_ids, prev_ids, app_df):
    print("Generating installments_payments.csv ...")
    # Payment behaviour correlated with default: defaulters pay less on time
    target_map = dict(zip(app_df["SK_ID_CURR"], app_df["TARGET"]))
    sample_pids = rng.choice(prev_ids, size=min(10_000, len(prev_ids)), replace=False)
    prev_df_tmp = pd.read_csv(os.path.join(DATA_DIR, "previous_application.csv"),
                               usecols=["SK_ID_PREV","SK_ID_CURR"])
    pid_to_sid  = dict(zip(prev_df_tmp["SK_ID_PREV"], prev_df_tmp["SK_ID_CURR"]))

    rows = []
    for pid in sample_pids:
        sid        = pid_to_sid.get(pid)
        if sid is None: continue
        is_default = target_map.get(sid, 0)

        n_inst   = int(rng.integers(6, 36))
        amt_due  = float(rng.lognormal(9.2, 0.55))

        # Defaulters: lower payment ratio, more late payments
        pay_alpha = 5.0 if not is_default else 2.5
        pay_beta  = 2.0 if not is_default else 3.0
        late_prob = 0.06 if not is_default else 0.22

        for inst_num in range(1, n_inst + 1):
            due_day    = -(n_inst - inst_num) * 30
            pay_ratio  = float(rng.beta(pay_alpha, pay_beta))
            late_days  = int(rng.integers(1, 45)) if rng.random() < late_prob else 0
            actual_day = due_day + late_days
            rows.append({
                "SK_ID_PREV":             pid,
                "SK_ID_CURR":             sid,
                "NUM_INSTALMENT_VERSION": 1.0,
                "NUM_INSTALMENT_NUMBER":  float(inst_num),
                "DAYS_INSTALMENT":        float(due_day),
                "DAYS_ENTRY_PAYMENT":     float(actual_day) if rng.random() > 0.02 else np.nan,
                "AMT_INSTALMENT":         amt_due,
                "AMT_PAYMENT":            amt_due * pay_ratio,
            })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(DATA_DIR, "installments_payments.csv"), index=False)
    print(f"  Saved. shape={df.shape}")


# ── 6. credit_card_balance.csv ───────────────────────────────────────────────
def make_credit_card(app_ids, prev_ids, app_df):
    print("Generating credit_card_balance.csv ...")
    target_map  = dict(zip(app_df["SK_ID_CURR"], app_df["TARGET"]))
    sample_pids = rng.choice(prev_ids, size=min(4_000, len(prev_ids)), replace=False)
    prev_df_tmp = pd.read_csv(os.path.join(DATA_DIR, "previous_application.csv"),
                               usecols=["SK_ID_PREV","SK_ID_CURR"])
    pid_to_sid  = dict(zip(prev_df_tmp["SK_ID_PREV"], prev_df_tmp["SK_ID_CURR"]))

    rows = []
    for pid in sample_pids:
        sid        = pid_to_sid.get(pid)
        if sid is None: continue
        is_default = target_map.get(sid, 0)

        limit     = float(rng.lognormal(11.2, 0.65))
        n_months  = int(rng.integers(3, 30))

        # Defaulters: higher utilisation, more DPD
        util_a    = 2.0 if not is_default else 4.0
        util_b    = 4.0 if not is_default else 2.0
        dpd_prob  = 0.04 if not is_default else 0.18

        for m in range(-n_months, 0):
            bal      = limit * float(rng.beta(util_a, util_b))
            drawing  = float(rng.lognormal(8, 0.8)) if rng.random() < 0.5 else 0.0
            min_pay  = bal * 0.05
            pay      = min_pay * float(rng.uniform(0.9, 3.0))
            dpd_val  = int(rng.integers(1, 60)) if rng.random() < dpd_prob else 0
            rows.append({
                "SK_ID_PREV": pid, "SK_ID_CURR": sid,
                "MONTHS_BALANCE":              m,
                "AMT_BALANCE":                 bal,
                "AMT_CREDIT_LIMIT_ACTUAL":     limit,
                "AMT_DRAWINGS_ATM_CURRENT":    drawing * 0.3,
                "AMT_DRAWINGS_CURRENT":        drawing,
                "AMT_DRAWINGS_OTHER_CURRENT":  0.0,
                "AMT_DRAWINGS_POS_CURRENT":    drawing * 0.7,
                "AMT_INST_MIN_REGULARITY":     min_pay,
                "AMT_PAYMENT_CURRENT":         pay,
                "AMT_PAYMENT_TOTAL_CURRENT":   pay,
                "AMT_RECEIVABLE_PRINCIPAL":    bal * 0.9,
                "AMT_RECIVABLE":               bal,
                "AMT_TOTAL_RECEIVABLE":        bal * 1.05,
                "CNT_DRAWINGS_ATM_CURRENT":    int(rng.integers(0, 5)),
                "CNT_DRAWINGS_CURRENT":        int(rng.integers(0, 10)),
                "CNT_DRAWINGS_OTHER_CURRENT":  0,
                "CNT_DRAWINGS_POS_CURRENT":    int(rng.integers(0, 8)),
                "CNT_INSTALMENT_MATURE_CUM":   float(rng.integers(0, 60)),
                "NAME_CONTRACT_STATUS":        rng.choice(["Active","Completed","Signed"], p=[0.70,0.25,0.05]),
                "SK_DPD":     dpd_val,
                "SK_DPD_DEF": max(0, dpd_val - 30),
            })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(DATA_DIR, "credit_card_balance.csv"), index=False)
    print(f"  Saved. shape={df.shape}")


# ── 7. POS_CASH_balance.csv ──────────────────────────────────────────────────
def make_pos_cash(app_ids, prev_ids):
    print("Generating POS_CASH_balance.csv ...")
    sample_pids = rng.choice(prev_ids, size=min(6_000, len(prev_ids)), replace=False)
    prev_df_tmp = pd.read_csv(os.path.join(DATA_DIR, "previous_application.csv"),
                               usecols=["SK_ID_PREV","SK_ID_CURR"])
    pid_to_sid  = dict(zip(prev_df_tmp["SK_ID_PREV"], prev_df_tmp["SK_ID_CURR"]))
    rows = []
    for pid in sample_pids:
        sid = pid_to_sid.get(pid)
        if sid is None: continue
        n_months   = int(rng.integers(3, 36))
        inst_total = int(rng.integers(6, 36))
        for m in range(-n_months, 0):
            rows.append({
                "SK_ID_PREV": pid, "SK_ID_CURR": sid,
                "MONTHS_BALANCE":          m,
                "CNT_INSTALMENT":          float(inst_total),
                "CNT_INSTALMENT_FUTURE":   max(0, inst_total + m),
                "NAME_CONTRACT_STATUS":    rng.choice(["Active","Completed","Signed"], p=[0.5,0.45,0.05]),
                "SK_DPD":     int(rng.integers(0,60)) if rng.random() < 0.07 else 0,
                "SK_DPD_DEF": int(rng.integers(0,30)) if rng.random() < 0.03 else 0,
            })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(DATA_DIR, "POS_CASH_balance.csv"), index=False)
    print(f"  Saved. shape={df.shape}")


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    print("="*60)
    print("Generating calibrated synthetic dataset v2")
    print(f"  n_applications = {N_APP:,}")
    print(f"  target_rate    = {TARGET_RATE:.1%}")
    print(f"  thin_file_rate = ~13.5%")
    print("="*60)

    app_ids, app_df = make_application()
    bur_ids         = make_bureau(app_ids, app_df)
    make_bureau_balance(bur_ids)
    prev_ids        = make_previous(app_ids)
    make_installments(app_ids, prev_ids, app_df)
    make_credit_card(app_ids, prev_ids, app_df)
    make_pos_cash(app_ids, prev_ids)

    print("\n" + "="*60)
    print("All files generated. Run:  python run_full_pipeline.py --skip-tuning")
    print("="*60)

if __name__ == "__main__":
    main()
