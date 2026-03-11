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
            return _role2_customer_insert(table, cols)
        elif verb == "UPDATE":
            return _role2_customer_update(table, cols)
        elif verb == "DELETE":
            return _role2_customer_delete(table, cols)
    elif table == "orders":
        if verb == "INSERT":
            return _role2_orders_insert(table, cols)
        elif verb == "UPDATE":
            return _role2_orders_update(table, cols)
        elif verb == "DELETE":
            return _role2_orders_delete(table, cols)
    elif table in ["sales", "leads"]:
        if verb == "INSERT":
            return _role2_insert(table, cols)
        elif verb == "UPDATE":
            return _role2_update(table, cols)
        elif verb == "DELETE":
            return _role2_delete(table, cols)
    return f"-- 未定义 {table} 的 {verb} 操作"

def _role2_customer_insert(table, cols):
    specific_cols = random.sample(['customer_id', 'cust_name', 'cust_level', 'last_purchase'], k=random.randint(1, 3))
    col_str = ", ".join(specific_cols)
    vals = ", ".join([f"'{table}_{c}_val_{random.randint(100, 999)}'" for c in specific_cols])
    return f"INSERT INTO {table} ({col_str}) VALUES ({vals})"

def _role2_customer_update(table, cols):
    set_clause = ", ".join([f"{c} = '{table}_{c}_updated_{random.randint(1, 100)}'" for c in cols])
    return (
        f"UPDATE {table} SET {set_clause}, last_modified = CURRENT_TIMESTAMP "
        f"WHERE {table}.customer_id = 'CUST_{random.randint(1000, 9999)}' "
        f"AND {table}.cust_level = 'VIP'"
    )

def _role2_customer_delete(table, cols):
    return (
        f"DELETE FROM {table} "
        f"WHERE {table}.customer_id = 'CUST_{random.randint(1000, 9999)}' "
        f"AND {table}.account_status = 'INACTIVE'"
    )

def _role2_customer_select(table, cols):
    col_str = ", ".join([f"{table}.{c}" for c in ["customer_id", "cust_name", "points", "last_purchase"]])
    return (
        f"SELECT {col_str} FROM {table} "
        f"WHERE {table}.cust_level = 'VIP' "
        f"AND {table}.points > 1000 "
        f"ORDER BY {table}.last_purchase DESC, {table}.points ASC"
    )

def _role2_orders_insert(table, cols):
    specific_cols = random.sample(['order_id', 'prod_name', 'qty', 'total_price'], k=random.randint(2, 4))
    col_str = ", ".join(specific_cols)
    vals = ", ".join([f"'{table}_{c}_val_{random.randint(100, 999)}'" for c in specific_cols])
    return f"INSERT INTO {table} ({col_str}) VALUES ({vals})"

def _role2_orders_update(table, cols):
    set_clause = ", ".join([f"{c} = '{table}_{c}_updated_{random.randint(1, 100)}'" for c in cols])
    return (
        f"UPDATE {table} SET {set_clause}, status = 'SHIPPED' "
        f"WHERE {table}.order_id IN (SELECT order_id FROM {table} WHERE status = 'PENDING') "
        f"AND {table}.total_price > 1000"
    )

def _role2_orders_delete(table, cols):
    return (
        f"DELETE FROM {table} "
        f"WHERE EXISTS (SELECT 1 FROM {table} AS sub WHERE sub.order_id = {table}.order_id AND sub.status = 'CANCELLED') "
        f"AND {table}.qty < 5"
    )

def _role2_orders_select(table, cols):
    col_str = ", ".join([f"{table}.{c}" for c in ["order_id", "prod_name", "qty", "total_price"]])
    return (
        f"SELECT {col_str}, SUM({table}.total_price) AS total_sales "
        f"FROM {table} "
        f"WHERE {table}.status = 'COMPLETED' "
        f"GROUP BY {col_str} "
        f"ORDER BY total_sales DESC, {table}.prod_name ASC"
    )

def _role2_insert(table, cols):
    specific_cols = random.sample(cols, k=random.randint(3, 4))
    col_str = ", ".join(specific_cols)
    vals = ", ".join([f"'{table}_{c}_val_{random.randint(100, 999)}'" for c in specific_cols])
    return f"INSERT INTO {table} ({col_str}) VALUES ({vals})"

def _role2_update(table, cols):
    set_clause = ", ".join([f"{c} = '{table}_{c}_updated_{random.randint(1, 100)}'" for c in cols])
    if table == "sales":
        return (
            f"UPDATE {table} SET {set_clause}, achieved = achieved + 100 "
            f"WHERE {table}.region_code IN (SELECT region_code FROM {table} WHERE monthly_target > 5000) "
            f"AND EXISTS (SELECT 1 FROM orders WHERE orders.order_id = {table}.sales_id AND orders.status = 'COMPLETED') "
            f"AND {table}.total_sales_value > 10000"
        )
    elif table == "leads":
        return (
            f"UPDATE {table} SET {set_clause}, lead_status = 'FOLLOW_UP' "
            f"WHERE {table}.intent = 'HIGH' "
            f"AND {table}.industry IN (SELECT industry FROM {table} WHERE lead_status = 'NEW') "
            f"AND EXISTS (SELECT 1 FROM customer WHERE customer.customer_id = {table}.leads_id AND customer.account_status = 'ACTIVE') "
            f"AND {table}.email_opt_in = TRUE"
        )

def _role2_delete(table, cols):
    if table == "sales":
        return (
            f"DELETE FROM {table} "
            f"WHERE {table}.sales_id IN (SELECT sales_id FROM {table} WHERE achieved < 1000) "
            f"AND NOT EXISTS (SELECT 1 FROM inventory WHERE inventory.item_id = {table}.sales_id) "
            f"AND {table}.region_code = 'REG_{random.randint(1, 10)}' "
            f"AND {table}.monthly_target > 2000"
        )
    elif table == "leads":
        return (
            f"DELETE FROM {table} "
            f"WHERE {table}.leads_id IN (SELECT leads_id FROM {table} WHERE lead_status = 'DISQUALIFIED') "
            f"AND {table}.source_web LIKE '%example.com%' "
            f"AND EXISTS (SELECT 1 FROM campaigns WHERE campaigns.camp_id = {table}.leads_id AND campaigns.budget_limit > 10000) "
            f"AND {table}.intent = 'LOW'"
        )

def _role2_sales_select(table, cols):
    col_str = ", ".join([f"{table}.{c}" for c in ["sales_id", "rep_id", "monthly_target", "achieved", "region_code"]])
    return (
        f"SELECT {col_str}, SUM({table}.achieved) AS total_achieved, AVG({table}.monthly_target) AS avg_target "
        f"FROM {table} "
        f"WHERE {table}.region_code IN (SELECT region_code FROM {table} WHERE total_sales_value > 10000) "
        f"AND {table}.monthly_target > 5000 "
        f"GROUP BY {col_str} "
        f"HAVING total_achieved > 10000 "
        f"ORDER BY total_achieved DESC, {table}.region_code ASC"
    )

def _role2_leads_select(table, cols):
    col_str = ", ".join([f"{table}.{c}" for c in ["leads_id", "company", "industry", "intent", "lead_status"]])
    return (
        f"SELECT {col_str}, COUNT(*) AS lead_count, MAX({table}.intent) AS max_intent "
        f"FROM {table} "
        f"WHERE {table}.industry IN (SELECT industry FROM {table} WHERE lead_status = 'NEW') "
        f"AND {table}.email_opt_in = TRUE "
        f"GROUP BY {col_str} "
        f"HAVING lead_count > 5 "
        f"ORDER BY max_intent DESC, {table}.company ASC"
    )

