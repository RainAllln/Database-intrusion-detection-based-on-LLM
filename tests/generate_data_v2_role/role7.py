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
            return _role7_customer_insert(table, cols)
        elif verb == "UPDATE":
            return _role7_customer_update(table, cols)
        elif verb == "DELETE":
            return _role7_customer_delete(table, cols)
    elif table == "orders":
        if verb == "INSERT":
            return _role7_orders_insert(table, cols)
        elif verb == "UPDATE":
            return _role7_orders_update(table, cols)
        elif verb == "DELETE":
            return _role7_orders_delete(table, cols)
    elif table == "tickets":
        if verb == "INSERT":
            return _role7_tickets_insert(table, cols)
        elif verb == "UPDATE":
            return _role7_tickets_update(table, cols)
        elif verb == "DELETE":
            return _role7_tickets_delete(table, cols)
        elif verb == "SELECT":
            return _role7_tickets_select(table, cols)
    return f"-- 未定义 {table} 的 {verb} 操作"

def _role7_customer_insert(table, cols):
    specific_cols = random.sample(['customer_id', 'account_status', 'home_address', 'credit_card'], k=random.randint(1, 3))
    col_str = ", ".join(specific_cols)
    vals = ", ".join([f"'{table}_{c}_val_{random.randint(100, 999)}'" for c in specific_cols])
    return f"INSERT INTO {table} ({col_str}) VALUES ({vals})"

def _role7_customer_update(table, cols):
    set_clause = ", ".join([f"{c} = '{table}_{c}_updated_{random.randint(1, 100)}'" for c in cols])
    return (
        f"UPDATE {table} SET {set_clause}, feedback = 'Updated by Customer Service' "
        f"WHERE {table}.customer_id = 'CUST_{random.randint(1000, 9999)}' "
        f"AND {table}.account_status = 'ACTIVE'"
    )

def _role7_customer_delete(table, cols):
    return (
        f"DELETE FROM {table} "
        f"WHERE {table}.customer_id = 'CUST_{random.randint(1000, 9999)}' "
        f"AND {table}.last_purchase < '2022-01-01' "
        f"AND {table}.account_status = 'CLOSED'"
    )

def _role7_customer_select(table, cols):
    col_str = ", ".join([f"{table}.{c}" for c in ["customer_id", "feedback", "account_status", "last_purchase"]])
    return (
        f"SELECT {col_str} FROM {table} "
        f"WHERE {table}.account_status = 'ACTIVE' "
        f"AND {table}.last_purchase > '2023-01-01' "
        f"ORDER BY {table}.last_purchase ASC, {table}.feedback DESC"
    )

def _role7_orders_insert(table, cols):
    specific_cols = random.sample(['order_id', 'prod_name', 'discount', 'fraud_check_score'], k=random.randint(2, 4))
    col_str = ", ".join(specific_cols)
    vals = ", ".join([f"'{table}_{c}_val_{random.randint(100, 999)}'" for c in specific_cols])
    return f"INSERT INTO {table} ({col_str}) VALUES ({vals})"

def _role7_orders_update(table, cols):
    set_clause = ", ".join([f"{c} = '{table}_{c}_updated_{random.randint(1, 100)}'" for c in cols])
    return (
        f"UPDATE {table} SET {set_clause}, feedback = 'Updated by Customer Service' "
        f"WHERE {table}.order_id IN (SELECT order_id FROM {table} WHERE fraud_check_score > 80) "
        f"AND {table}.status = 'DELIVERED'"
    )

def _role7_orders_delete(table, cols):
    return (
        f"DELETE FROM {table} "
        f"WHERE {table}.order_id IN (SELECT order_id FROM {table} WHERE fraud_check_score < 50) "
        f"AND {table}.discount > 10"
    )

def _role7_orders_select(table, cols):
    col_str = ", ".join([f"{table}.{c}" for c in ["order_id", "prod_name", "discount", "feedback"]])
    return (
        f"SELECT {col_str} "
        f"FROM {table} "
        f"WHERE {table}.status = 'DELIVERED' "
        f"AND {table}.discount > (SELECT AVG(discount) FROM {table} WHERE status = 'DELIVERED') "
        f"ORDER BY {table}.feedback DESC, {table}.prod_name ASC"
    )

def _role7_tickets_insert(table, cols):
    specific_cols = random.sample(['ticket_id', 'issue_type', 'status', 'priority', 'assigned_to'], k=random.randint(4, 5))
    col_str = ", ".join(specific_cols)
    vals = ", ".join([f"'{table}_{c}_val_{random.randint(100, 999)}'" for c in specific_cols])
    return f"INSERT INTO {table} ({col_str}) VALUES ({vals})"

def _role7_tickets_update(table, cols):
    set_clause = ", ".join([f"{c} = '{table}_{c}_updated_{random.randint(1, 100)}'" for c in cols])
    return (
        f"UPDATE {table} SET {set_clause}, ticket_closed_date = CURRENT_TIMESTAMP, resolution_notes = 'Resolved by Customer Service' "
        f"WHERE {table}.status = 'OPEN' "
        f"AND {table}.priority = 'HIGH' "
        f"AND {table}.assigned_to IN (SELECT emp_id FROM employee WHERE position = 'Support Agent')"
    )

def _role7_tickets_delete(table, cols):
    return (
        f"DELETE FROM {table} "
        f"WHERE {table}.ticket_id IN (SELECT ticket_id FROM {table} WHERE status = 'CLOSED' AND feedback IS NULL) "
        f"AND {table}.priority = 'LOW' "
        f"AND {table}.ticket_opened_date < CURRENT_DATE - INTERVAL '1 year'"
    )

def _role7_tickets_select(table, cols):
    col_str = ", ".join([f"{table}.{c}" for c in cols])
    return (
        f"SELECT {col_str}, resolution_notes, feedback "
        f"FROM {table} "
        f"WHERE {table}.status = 'CLOSED' "
        f"AND {table}.feedback IS NOT NULL "
        f"UNION ALL "
        f"SELECT {col_str}, 'No resolution' AS resolution_notes, 'No feedback' AS feedback "
        f"FROM {table} "
        f"WHERE {table}.status = 'OPEN' "
        f"AND {table}.priority = 'HIGH' "
        f"ORDER BY feedback DESC, resolution_notes ASC"
    )

