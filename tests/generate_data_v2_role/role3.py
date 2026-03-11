import sys
import os
# 将根目录添加到sys.path中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ..generate_data_v2 import SCHEMA
import random

def build_sql(table, verb, cols):
    cols = SCHEMA[table]  # 使用表的全部列
    if table == "dev" and verb == "SELECT":
        return _role3_dev_select(table, cols)
    elif table == "logs" and verb == "SELECT":
        return _role3_logs_select(table, cols)
    elif table == "config" and verb == "SELECT":
        return _role3_config_select(table, cols)
    elif table == "test_table" and verb == "SELECT":
        return _role3_test_table_select(table, cols)
    if table == "dev":
        if verb == "INSERT":
            return _role3_dev_insert(table, cols)
        elif verb == "UPDATE":
            return _role3_dev_update(table, cols)
        elif verb == "DELETE":
            return _role3_dev_delete(table, cols)
    elif table == "logs":
        if verb == "INSERT":
            return _role3_logs_insert(table, cols)
        elif verb == "UPDATE":
            return _role3_logs_update(table, cols)
        elif verb == "DELETE":
            return _role3_logs_delete(table, cols)
    elif table == "config":
        if verb == "INSERT":
            return _role3_config_insert(table, cols)
        elif verb == "UPDATE":
            return _role3_config_update(table, cols)
        elif verb == "DELETE":
            return _role3_config_delete(table, cols)
    elif table == "test_table":
        if verb == "INSERT":
            return _role3_test_table_insert(table, cols)
        elif verb == "UPDATE":
            return _role3_test_table_update(table, cols)
        elif verb == "DELETE":
            return _role3_test_table_delete(table, cols)
    return f"-- 未定义 {table} 的 {verb} 操作"

def _role3_dev_insert(table, cols):
    specific_cols = random.sample(SCHEMA[table], k=6)
    col_str = ", ".join(specific_cols)
    vals = ", ".join([f"'dev_{c}_val_{random.randint(100, 999)}'" for c in specific_cols])
    return f"INSERT INTO {table} ({col_str}) VALUES ({vals})"

def _role3_dev_update(table, cols):
    set_clause = ", ".join([f"{c} = 'updated_{random.randint(1, 100)}'" for c in cols])
    return (
        f"UPDATE {table} SET {set_clause}, deployment_date = CURRENT_TIMESTAMP "
        f"WHERE {table}.dev_id = 'DEV_{random.randint(1000, 9999)}' "
        f"AND {table}.version > 1.0 "
        f"AND {table}.branch LIKE 'feature/%'"
    )

def _role3_dev_delete(table, cols):
    return (
        f"DELETE FROM {table} "
        f"WHERE {table}.dev_id IN (SELECT dev_id FROM {table} WHERE version < 1.0) "
        f"AND {table}.test_environment = 'staging' "
        f"AND {table}.deploy_id IS NOT NULL"
    )

def _role3_dev_select(table, cols):
    col_str = ", ".join([f"{table}.{c}" for c in cols])
    return (
        f"SELECT {col_str} FROM {table} "
        f"WHERE {table}.test_environment = 'production' "
        f"AND {table}.version > 1.0 "
        f"ORDER BY {table}.deployment_date DESC"
    )

def _role3_logs_insert(table, cols):
    specific_cols = random.sample(SCHEMA[table], k=7)
    col_str = ", ".join(specific_cols)
    vals = ", ".join([f"'log_{c}_val_{random.randint(100, 999)}'" for c in specific_cols])
    return f"INSERT INTO {table} ({col_str}) VALUES ({vals})"

def _role3_logs_update(table, cols):
    set_clause = ", ".join([f"{c} = 'updated_{random.randint(1, 100)}'" for c in cols])
    return (
        f"UPDATE {table} SET {set_clause}, timestamp = CURRENT_TIMESTAMP "
        f"WHERE {table}.log_id = 'LOG_{random.randint(1000, 9999)}' "
        f"AND {table}.level IN ('ERROR', 'WARN') "
        f"AND {table}.service LIKE 'service_%'"
    )

def _role3_logs_delete(table, cols):
    return (
        f"DELETE FROM {table} "
        f"WHERE {table}.log_id IN (SELECT log_id FROM {table} WHERE level = 'DEBUG') "
        f"AND {table}.timestamp < NOW() - INTERVAL '30 days' "
        f"AND {table}.ip_addr IS NOT NULL"
    )

def _role3_logs_select(table, cols):
    col_str = ", ".join([f"{table}.{c}" for c in cols])
    return (
        f"SELECT {col_str} FROM {table} "
        f"WHERE {table}.level IN ('INFO', 'ERROR') "
        f"AND {table}.timestamp > NOW() - INTERVAL '7 days' "
        f"ORDER BY {table}.timestamp DESC"
    )

def _role3_config_insert(table, cols):
    specific_cols = random.sample(SCHEMA[table], k=6)
    col_str = ", ".join(specific_cols)
    vals = ", ".join([f"'config_{c}_val_{random.randint(100, 999)}'" for c in specific_cols])
    return f"INSERT INTO {table} ({col_str}) VALUES ({vals})"

def _role3_config_update(table, cols):
    set_clause = ", ".join([f"{c} = 'updated_{random.randint(1, 100)}'" for c in cols])
    return (
        f"UPDATE {table} SET {set_clause}, last_modified = CURRENT_TIMESTAMP "
        f"WHERE {table}.config_id = 'CONF_{random.randint(1000, 9999)}' "
        f"AND {table}.is_active = TRUE "
        f"AND {table}.env = 'production'"
    )

def _role3_config_delete(table, cols):
    return (
        f"DELETE FROM {table} "
        f"WHERE {table}.config_id IN (SELECT config_id FROM {table} WHERE is_active = FALSE) "
        f"AND {table}.config_type = 'deprecated' "
        f"AND {table}.version < 2.0"
    )

def _role3_config_select(table, cols):
    col_str = ", ".join([f"{table}.{c}" for c in cols])
    return (
        f"SELECT {col_str} FROM {table} "
        f"WHERE {table}.is_active = TRUE "
        f"AND {table}.env = 'staging' "
        f"ORDER BY {table}.last_modified DESC"
    )

def _role3_test_table_insert(table, cols):
    specific_cols = random.sample(SCHEMA[table], k=7)
    col_str = ", ".join(specific_cols)
    vals = ", ".join([f"'test_{c}_val_{random.randint(100, 999)}'" for c in specific_cols])
    return f"INSERT INTO {table} ({col_str}) VALUES ({vals})"

def _role3_test_table_update(table, cols):
    set_clause = ", ".join([f"{c} = 'updated_{random.randint(1, 100)}'" for c in cols])
    return (
        f"UPDATE {table} SET {set_clause}, updated_at = CURRENT_TIMESTAMP "
        f"WHERE {table}.test_id = 'TEST_{random.randint(1000, 9999)}' "
        f"AND {table}.test_status = 'FAILED' "
        f"AND {table}.test_duration > 100"
    )

def _role3_test_table_delete(table, cols):
    return (
        f"DELETE FROM {table} "
        f"WHERE {table}.test_id IN (SELECT test_id FROM {table} WHERE test_status = 'OBSOLETE') "
        f"AND {table}.error_message IS NOT NULL "
        f"AND {table}.test_result = 'ERROR'"
    )

def _role3_test_table_select(table, cols):
    col_str = ", ".join([f"{table}.{c}" for c in cols])
    return (
        f"SELECT {col_str} FROM {table} "
        f"WHERE {table}.test_status = 'PASSED' "
        f"AND {table}.test_duration < 300 "
        f"ORDER BY {table}.updated_at DESC"
    )

