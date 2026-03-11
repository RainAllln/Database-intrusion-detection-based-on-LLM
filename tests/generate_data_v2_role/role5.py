import sys
import os
# 将根目录添加到sys.path中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ..generate_data_v2 import _safe_val, SCHEMA
import random

def build_sql(table, verb, cols):
    cols = SCHEMA[table]  # 使用表的全部列
    if table == "customer":
        if verb == "INSERT":
            return _role5_customer_insert(table, cols)
        elif verb == "UPDATE":
            return _role5_customer_update(table, cols)
        elif verb == "DELETE":
            return _role5_customer_delete(table, cols)
    elif table == "revenue":
        if verb == "SELECT":
            return _role5_revenue_select(table, cols)
        elif verb == "INSERT":
            return _role5_revenue_insert(table, cols)
        elif verb == "UPDATE":
            return _role5_revenue_update(table, cols)
        elif verb == "DELETE":
            return _role5_revenue_delete(table, cols)
    elif table == "campaigns":
        if verb == "INSERT":
            return _role5_campaigns_insert(table, cols)
        elif verb == "UPDATE":
            return _role5_campaigns_update(table, cols)
        elif verb == "DELETE":
            return _role5_campaigns_delete(table, cols)
        elif verb == "SELECT":
            return _role5_campaigns_select(table, cols)
    return f"-- 未定义 {table} 的 {verb} 操作"

def _role5_revenue_insert(table, cols):
    col_str = ", ".join(cols)
    vals = ", ".join([f"'{table}_{c}_val_{random.randint(100, 999)}'" for c in cols])
    return (
        f"INSERT INTO {table} ({col_str}, created_at, created_by, campaign_id) "
        f"VALUES ({vals}, CURRENT_TIMESTAMP, 'MARKETING_USER', 'CAMP_{random.randint(1000, 9999)}') "
        f"ON DUPLICATE KEY UPDATE {cols[0]} = VALUES({cols[0]}), last_modified = CURRENT_TIMESTAMP"
    )

def _role5_revenue_update(table, cols):
    set_clause = ", ".join([f"{c} = '{table}_{c}_updated_{random.randint(1, 100)}'" for c in cols[1:]])
    return (
        f"UPDATE {table} SET {set_clause}, last_modified = CURRENT_TIMESTAMP, campaign_id = 'CAMP_{random.randint(1000, 9999)}' "
        f"WHERE {cols[0]} = '{table}_key_{random.randint(1, 100)}' "
        f"AND status = 'PENDING' AND version > 0"
    )

def _role5_revenue_delete(table, cols):
    return (
        f"DELETE FROM {table} "
        f"WHERE {cols[0]} = '{table}_delete_{random.randint(1, 100)}' "
        f"AND {table}.region = 'West' AND campaign_id = 'CAMP_{random.randint(1000, 9999)}'"
    )

def _role5_revenue_select(table, cols):
    col_str = ", ".join([f"{table}.{c}" for c in ["rep_id", "channel", "gross_income", "campaign_id"]])
    return (
        f"SELECT {col_str} FROM {table} "
        f"WHERE {table}.channel IN ('Online', 'Offline') "
        f"AND {table}.gross_income > 10000 "
        f"AND {table}.campaign_id IS NOT NULL "
        f"GROUP BY {table}.channel, {table}.campaign_id "
        f"ORDER BY {table}.gross_income DESC"
    )

def _role5_customer_insert(table, cols):
    specific_cols = random.sample(['customer_id', 'points', 'public_phone', 'email'], k=random.randint(1, 3))
    col_str = ", ".join(specific_cols)
    vals = ", ".join([f"'{table}_{c}_val_{random.randint(100, 999)}'" for c in specific_cols])
    return f"INSERT INTO {table} ({col_str}) VALUES ({vals})"

def _role5_customer_update(table, cols):
    set_clause = ", ".join([f"{c} = '{table}_{c}_updated_{random.randint(1, 100)}'" for c in cols])
    return (
        f"UPDATE {table} SET {set_clause}, campaign_id = 'CAMP_{random.randint(1000, 9999)}' "
        f"WHERE {table}.customer_id = 'CUST_{random.randint(1000, 9999)}' "
        f"AND {table}.cust_level IN ('Gold', 'Platinum')"
    )

def _role5_customer_delete(table, cols):
    return (
        f"DELETE FROM {table} "
        f"WHERE {table}.customer_id = 'CUST_{random.randint(1000, 9999)}' "
        f"AND {table}.points < 1000 "
        f"AND {table}.cust_level = 'Silver'"
    )

def _role5_customer_select(table, cols):
    col_str = ", ".join([f"{table}.{c}" for c in ["customer_id", "email", "public_phone", "campaign_id"]])
    return (
        f"SELECT {col_str} FROM {table} "
        f"WHERE {table}.campaign_id IS NOT NULL "
        f"AND {table}.cust_level IN ('Gold', 'Platinum') "
        f"ORDER BY {table}.campaign_id ASC, {table}.email DESC"
    )

def _role5_campaigns_insert(table, cols):
    specific_cols = random.sample(['camp_id', 'camp_name', 'budget_limit', 'channel', 'start_date'], k=random.randint(4, 5))
    col_str = ", ".join(specific_cols)
    vals = ", ".join([f"'{table}_{c}_val_{random.randint(100, 999)}'" for c in specific_cols])
    return f"INSERT INTO {table} ({col_str}) VALUES ({vals})"

def _role5_campaigns_update(table, cols):
    set_clause = ", ".join([f"{c} = '{table}_{c}_updated_{random.randint(1, 100)}'" for c in cols])
    return (
        f"UPDATE {table} SET {set_clause}, last_modified = CURRENT_TIMESTAMP, roi = roi + 0.1 "
        f"WHERE {table}.camp_id = 'CAMP_{random.randint(1000, 9999)}' "
        f"AND {table}.actual_spend > {table}.budget_limit "
        f"AND {table}.conversion_rate < 0.05"
    )

def _role5_campaigns_delete(table, cols):
    return (
        f"DELETE FROM {table} "
        f"WHERE {table}.camp_id IN (SELECT camp_id FROM {table} WHERE end_date < CURRENT_DATE) "
        f"AND {table}.roi < 1.0 "
        f"AND {table}.target_audience = 'General Public'"
    )

def _role5_campaigns_select(table, cols):
    col_str = ", ".join([f"{table}.{c}" for c in cols])
    return (
        f"SELECT {col_str}, COUNT(customer.customer_id) AS total_customers, AVG(revenue.net_profit) AS avg_profit "
        f"FROM {table} "
        f"LEFT JOIN customer ON {table}.camp_id = customer.customer_id "
        f"LEFT JOIN revenue ON {table}.camp_id = revenue.rep_id "
        f"WHERE {table}.channel IN ('Online', 'Offline') "
        f"AND {table}.budget_limit > 10000 "
        f"AND {table}.start_date >= CURRENT_DATE - INTERVAL '1 year' "
        f"GROUP BY {col_str} "
        f"HAVING total_customers > 50 "
        f"ORDER BY avg_profit DESC, {table}.start_date ASC"
    )

