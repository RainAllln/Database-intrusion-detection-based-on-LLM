import sys
import os
# 将根目录添加到sys.path中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ..generate_data_v2 import _safe_val, SCHEMA
import random

def build_sql(table, verb, cols):
    cols = SCHEMA[table]  # 使用表的全部列
    if table in ["finance", "salary", "budget"]:
        if verb == "INSERT":
            return _role0_finance_salary_budget_insert(table, cols)
        elif verb == "UPDATE":
            return _role0_finance_salary_budget_update(table, cols)
        elif verb == "DELETE":
            return _role0_finance_salary_budget_delete(table, cols)
        elif verb == "SELECT":
            return _role0_finance_salary_budget_select(table, cols)
    elif table == "revenue":
        if verb == "INSERT":
            return _role0_revenue_insert(table, cols)
        elif verb == "UPDATE":
            return _role0_revenue_update(table, cols)
        elif verb == "DELETE":
            return _role0_revenue_delete(table, cols)
        elif verb == "SELECT":
            return _role0_revenue_select(table, cols)
    return f"-- 未定义 {table} 的 {verb} 操作"

def _role0_finance_salary_budget_insert(table, cols):
    col_str = ", ".join(cols)
    vals = ", ".join([f"'{table}_{c}_val_{random.randint(100, 999)}'" for c in cols])
    return (
        f"INSERT INTO {table} ({col_str}, created_at, created_by) "
        f"VALUES ({vals}, CURRENT_TIMESTAMP, 'SYSTEM_USER') "
        f"ON DUPLICATE KEY UPDATE {cols[0]} = VALUES({cols[0]})"
    )

def _role0_finance_salary_budget_update(table, cols):
    set_clause = ", ".join([f"{c} = '{table}_{c}_updated_{random.randint(1, 100)}'" for c in cols[1:]])
    return (
        f"UPDATE {table} SET {set_clause}, updated_at = CURRENT_TIMESTAMP "
        f"WHERE {cols[0]} = '{table}_key_{random.randint(1, 100)}' "
        f"AND status = 'ACTIVE' AND version > 1"
    )

def _role0_finance_salary_budget_delete(table, cols):
    return (
        f"DELETE FROM {table} "
        f"WHERE {cols[0]} = '{table}_delete_{random.randint(1, 100)}' "
        f"AND EXISTS (SELECT 1 FROM audit_logs WHERE audit_logs.record_id = {table}.{cols[0]}) "
        f"AND audit_trace_log IS NOT NULL"
    )

def _role0_finance_salary_budget_select(table, cols):
    col_str = ", ".join([f"{table}.{c}" for c in cols])
    date_col = next((c for c in SCHEMA[table] if "date" in c), cols[0])
    amount_col = next((c for c in SCHEMA[table] if "amount" in c or "income" in c), cols[0])
    return (
        f"SELECT {col_str} FROM {table} "
        f"WHERE {table}.{date_col} >= '2023-01-01' AND {table}.{date_col} < '2026-01-01' "
        f"AND {table}.{amount_col} > 0 "
        f"AND {table}.category IN ('A', 'B', 'C') "
        f"ORDER BY {table}.{date_col} DESC, {table}.{amount_col} ASC"
    )

def _role0_revenue_insert(table, cols):
    col_str = ", ".join(cols)
    vals = ", ".join([f"'{table}_{c}_val_{random.randint(100, 999)}'" for c in cols])
    return (
        f"INSERT INTO {table} ({col_str}, created_at, created_by, approved_by) "
        f"VALUES ({vals}, CURRENT_TIMESTAMP, 'FINANCE_USER', 'MANAGER') "
        f"ON DUPLICATE KEY UPDATE {cols[0]} = VALUES({cols[0]}), updated_at = CURRENT_TIMESTAMP"
    )

def _role0_revenue_update(table, cols):
    set_clause = ", ".join([f"{c} = '{table}_{c}_updated_{random.randint(1, 100)}'" for c in cols[1:]])
    return (
        f"UPDATE {table} SET {set_clause}, updated_at = CURRENT_TIMESTAMP, approved_by = 'MANAGER' "
        f"WHERE {cols[0]} = '{table}_key_{random.randint(1, 100)}' "
        f"AND status = 'APPROVED' AND version > 1"
    )

def _role0_revenue_delete(table, cols):
    return (
        f"DELETE FROM {table} "
        f"WHERE {cols[0]} = '{table}_delete_{random.randint(1, 100)}' "
        f"AND EXISTS (SELECT 1 FROM audit_logs WHERE audit_logs.record_id = {table}.{cols[0]}) "
        f"AND audit_trace_log IS NOT NULL AND approved_by = 'MANAGER'"
    )

def _role0_revenue_select(table, cols):
    col_str = ", ".join([f"{table}.{c}" for c in ["rep_id", "month", "gross_revenue", "net_income"]])
    return (
        f"SELECT {col_str} FROM {table} "
        f"WHERE {table}.gross_revenue > 20000 "
        f"AND {table}.net_income > 10000 "
        f"AND {table}.month BETWEEN '2023-01' AND '2023-12' "
        f"ORDER BY {table}.month ASC, {table}.gross_revenue DESC"
    )

