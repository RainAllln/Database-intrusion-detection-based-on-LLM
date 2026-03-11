import sys
import os
# 将根目录添加到sys.path中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ..generate_data_v2 import _safe_val, SCHEMA
import random

def build_sql(table, verb, cols):
    cols = SCHEMA[table]  # 使用表的全部列
    if table in ["employee", "staff", "hr", "attendance", "performance"]:
        if verb == "INSERT":
            return _role1_common_insert(table, cols)
        elif verb == "UPDATE":
            return _role1_common_update(table, cols)
        elif verb == "DELETE":
            return _role1_common_delete(table, cols)
        elif verb == "SELECT":
            return _role1_common_select(table, cols)
    return f"-- 未定义 {table} 的 {verb} 操作"

def _role1_common_insert(table, cols):
    col_str = ", ".join(cols)
    vals = ", ".join([f"'{table}_{c}_val_{random.randint(100, 999)}'" for c in cols])
    return (
        f"INSERT INTO {table} ({col_str}, created_at, created_by) "
        f"VALUES ({vals}, CURRENT_TIMESTAMP, 'HR_USER') "
        f"ON DUPLICATE KEY UPDATE {cols[0]} = VALUES({cols[0]}), updated_at = CURRENT_TIMESTAMP"
    )

def _role1_common_update(table, cols):
    set_clause = ", ".join([f"{c} = '{table}_{c}_updated_{random.randint(1, 100)}'" for c in cols[1:]])
    return (
        f"UPDATE {table} SET {set_clause}, updated_at = CURRENT_TIMESTAMP, updated_by = 'HR_USER' "
        f"WHERE {cols[0]} = '{table}_key_{random.randint(1, 100)}' "
        f"AND status = 'ACTIVE' AND version > 1"
    )

def _role1_common_delete(table, cols):
    return (
        f"DELETE FROM {table} "
        f"WHERE {cols[0]} = '{table}_delete_{random.randint(1, 100)}' "
        f"AND EXISTS (SELECT 1 FROM audit_logs WHERE audit_logs.record_id = {table}.{cols[0]}) "
        f"AND audit_trace_log IS NOT NULL"
    )

def _role1_common_select(table, cols):
    col_str = ", ".join([f"{table}.{c}" for c in cols])
    return (
        f"SELECT {col_str} FROM {table} "
        f"WHERE {table}.status = 'ACTIVE' "
        f"AND {table}.{cols[0]} LIKE 'HR%' "
        f"ORDER BY {table}.entry_date DESC, {table}.rank ASC"
    )

