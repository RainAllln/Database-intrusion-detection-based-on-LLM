import sys
import os
# 将根目录添加到sys.path中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ..generate_data_v2 import _safe_val, SCHEMA

def build_sql(table, verb, cols):
    cols = SCHEMA[table]  # 使用表的全部列
    if table in ["finance", "salary", "budget"]:
        if verb == "SELECT":
            return _role6_finance_salary_budget_select(table, cols)
    elif table in ["employee", "staff", "hr", "attendance", "performance"]:
        if verb == "SELECT":
            return _role6_employee_staff_hr_attendance_performance_select(table, cols)
    elif table == "revenue":
        if verb == "SELECT":
            return _role6_revenue_select(table, cols)
    elif table == "sales":
        if verb == "SELECT":
            return _role6_sales_select(table, cols)
    elif table == "leads":
        if verb == "SELECT":
            return _role6_leads_select(table, cols)
    elif table == "dev" and verb == "SELECT":
        return _role6_dev_select(table, cols)
    elif table == "logs" and verb == "SELECT":
        return _role6_logs_select(table, cols)
    elif table == "config" and verb == "SELECT":
        return _role6_config_select(table, cols)
    elif table == "test_table" and verb == "SELECT":
        return _role6_test_table_select(table, cols)
    elif table == "inventory" and verb == "SELECT":
        return _role6_inventory_select(table, cols)
    elif table == "suppliers" and verb == "SELECT":
        return _role6_suppliers_select(table, cols)
    elif table == "campaigns" and verb == "SELECT":
        return _role6_campaigns_select(table, cols)
    elif table == "tickets" and verb == "SELECT":
        return _role6_tickets_select(table, cols)
    return f"-- 未定义 {table} 的 {verb} 操作"

def _role6_finance_salary_budget_select(table, cols):
    col_str = ", ".join([f"{table}.{c}" for c in cols])
    audit_col = next((c for c in SCHEMA[table] if "audit" in c or "trace" in c), cols[0])
    return (
        f"SELECT {col_str} FROM {table} "
        f"WHERE {table}.{audit_col} IS NOT NULL "
        f"AND {table}.hidden_assets = FALSE "
        f"AND {table}.{cols[0]} LIKE 'A%' "
        f"ORDER BY {table}.{cols[0]} ASC, {table}.{cols[-1]} DESC"
    )

def _role6_revenue_select(table, cols):
    col_str = ", ".join([f"{table}.{c}" for c in ["rep_id", "region", "net_profit", "audit_trace_log"]])
    return (
        f"SELECT {col_str} FROM {table} "
        f"WHERE {table}.audit_trace_log IS NOT NULL "
        f"AND {table}.net_profit > 5000 "
        f"AND {table}.region IN ('East', 'West') "
        f"ORDER BY {table}.region ASC, {table}.net_profit DESC"
    )

def _role6_employee_staff_hr_attendance_performance_select(table, cols):
    col_str = ", ".join([f"{table}.{c}" for c in cols])
    audit_col = next((c for c in SCHEMA[table] if "audit" in c or "trace" in c), cols[0])
    return (
        f"SELECT {col_str} FROM {table} "
        f"WHERE {table}.{audit_col} IS NOT NULL "
        f"AND {table}.status = 'VERIFIED' "
        f"AND {table}.{cols[0]} LIKE 'AUD%' "
        f"ORDER BY {table}.entry_date ASC, {table}.rank DESC"
    )

def _role6_customer_select(table, cols):
    col_str = ", ".join([f"{table}.{c}" for c in ["customer_id", "account_status", "audit_trace_log", "home_address"]])
    return (
        f"SELECT {col_str} FROM {table} "
        f"WHERE {table}.audit_trace_log IS NOT NULL "
        f"AND {table}.account_status = 'VERIFIED' "
        f"ORDER BY {table}.customer_id ASC, {table}.account_status DESC"
    )

def _role6_orders_select(table, cols):
    col_str = ", ".join([f"{table}.{c}" for c in ["order_id", "status", "total_price", "fraud_check_score"]])
    return (
        f"SELECT {col_str} "
        f"FROM {table} "
        f"WHERE {table}.fraud_check_score > 70 "
        f"AND {table}.status IN ('PENDING', 'PROCESSING') "
        f"AND EXISTS (SELECT 1 FROM customer WHERE customer.customer_id = {table}.order_id AND customer.account_status = 'VERIFIED') "
        f"ORDER BY {table}.total_price DESC"
    )

def _role6_sales_select(table, cols):
    col_str = ", ".join([f"{table}.{c}" for c in ["sales_id", "rep_id", "region_code", "total_sales_value", "achieved"]])
    return (
        f"SELECT {col_str}, COUNT(*) AS sales_count, MAX({table}.achieved) AS max_achieved "
        f"FROM {table} "
        f"LEFT JOIN orders ON {table}.sales_id = orders.order_id "
        f"WHERE {table}.total_sales_value > 5000 "
        f"AND EXISTS (SELECT 1 FROM revenue WHERE revenue.region = {table}.region_code AND revenue.net_profit > 10000) "
        f"GROUP BY {col_str} "
        f"HAVING sales_count > 3 "
        f"ORDER BY max_achieved DESC, {table}.region_code ASC"
    )

def _role6_leads_select(table, cols):
    col_str = ", ".join([f"{table}.{c}" for c in ["leads_id", "company", "industry", "intent", "lead_status"]])
    return (
        f"SELECT {col_str}, AVG({table}.intent) AS avg_intent, COUNT(*) AS total_leads "
        f"FROM {table} "
        f"INNER JOIN campaigns ON {table}.leads_id = campaigns.camp_id "
        f"WHERE {table}.lead_status = 'FOLLOW_UP' "
        f"AND {table}.source_web LIKE '%example.com%' "
        f"AND EXISTS (SELECT 1 FROM customer WHERE customer.customer_id = {table}.leads_id AND customer.account_status = 'VERIFIED') "
        f"GROUP BY {col_str} "
        f"HAVING total_leads > 10 "
        f"ORDER BY avg_intent DESC, {table}.company ASC"
    )

def _role6_dev_select(table, cols):
    col_str = ", ".join([f"{table}.{c}" for c in cols])
    return (
        f"SELECT {col_str}, COUNT(*) AS usage_count "
        f"FROM {table} "
        f"LEFT JOIN logs ON {table}.dev_id = logs.log_id "
        f"WHERE {table}.version > 2.0 "
        f"AND logs.level = 'ERROR' "
        f"GROUP BY {col_str} "
        f"HAVING usage_count > 5 "
        f"ORDER BY usage_count DESC"
    )

def _role6_logs_select(table, cols):
    col_str = ", ".join([f"{table}.{c}" for c in cols])
    return (
        f"SELECT {col_str}, MAX(timestamp) AS last_error_time "
        f"FROM {table} "
        f"WHERE {table}.level = 'ERROR' "
        f"AND EXISTS (SELECT 1 FROM dev WHERE dev.dev_id = {table}.log_id AND dev.test_environment = 'production') "
        f"GROUP BY {col_str} "
        f"ORDER BY last_error_time DESC"
    )

def _role6_config_select(table, cols):
    col_str = ", ".join([f"{table}.{c}" for c in cols])
    return (
        f"SELECT {col_str}, COUNT(*) AS active_count "
        f"FROM {table} "
        f"WHERE {table}.is_active = TRUE "
        f"AND {table}.env IN ('production', 'staging') "
        f"GROUP BY {col_str} "
        f"HAVING active_count > 10 "
        f"ORDER BY active_count DESC"
    )

def _role6_test_table_select(table, cols):
    col_str = ", ".join([f"{table}.{c}" for c in cols])
    return (
        f"SELECT {col_str}, AVG(test_duration) AS avg_duration "
        f"FROM {table} "
        f"WHERE {table}.test_status = 'FAILED' "
        f"AND {table}.error_message IS NOT NULL "
        f"GROUP BY {col_str} "
        f"HAVING avg_duration > 200 "
        f"ORDER BY avg_duration DESC"
    )

def _role6_inventory_select(table, cols):
    col_str = ", ".join([f"{table}.{c}" for c in cols])
    return (
        f"SELECT {col_str}, MAX(stock_qty) AS max_stock, MIN(stock_qty) AS min_stock "
        f"FROM {table} "
        f"WHERE {table}.stock_qty > 0 "
        f"AND EXISTS (SELECT 1 FROM suppliers WHERE suppliers.sup_id = {table}.sup_id AND suppliers.rating > 4) "
        f"GROUP BY {col_str} "
        f"HAVING max_stock > 100 "
        f"ORDER BY min_stock ASC, max_stock DESC"
    )

def _role6_suppliers_select(table, cols):
    col_str = ", ".join([f"{table}.{c}" for c in cols])
    return (
        f"SELECT {col_str}, COUNT(*) AS active_suppliers "
        f"FROM {table} "
        f"WHERE {table}.supplier_status = 'ACTIVE' "
        f"AND {table}.rating >= 3 "
        f"UNION ALL "
        f"SELECT {col_str}, COUNT(*) AS inactive_suppliers "
        f"FROM {table} "
        f"WHERE {table}.supplier_status = 'INACTIVE' "
        f"ORDER BY active_suppliers DESC"
    )

def _role6_campaigns_select(table, cols):
    col_str = ", ".join([f"{table}.{c}" for c in cols])
    return (
        f"SELECT {col_str}, MAX(actual_spend) AS max_spend, MIN(conversion_rate) AS min_conversion "
        f"FROM {table} "
        f"WHERE {table}.roi > 1.5 "
        f"AND EXISTS (SELECT 1 FROM revenue WHERE revenue.rep_id = {table}.camp_id AND revenue.net_profit > 5000) "
        f"GROUP BY {col_str} "
        f"HAVING max_spend > 20000 "
        f"UNION ALL "
        f"SELECT {col_str}, 0 AS max_spend, 0 AS min_conversion "
        f"FROM {table} "
        f"WHERE {table}.target_audience = 'Niche Market' "
        f"ORDER BY max_spend DESC, min_conversion ASC"
    )

def _role6_tickets_select(table, cols):
    col_str = ", ".join([f"{table}.{c}" for c in cols])
    return (
        f"SELECT {col_str}, COUNT(*) AS total_tickets, MAX(ticket_opened_date) AS last_opened "
        f"FROM {table} "
        f"LEFT JOIN customer ON {table}.user_id = customer.customer_id "
        f"WHERE {table}.status IN ('OPEN', 'IN_PROGRESS') "
        f"AND EXISTS (SELECT 1 FROM employee WHERE employee.emp_id = {table}.assigned_to AND employee.position = 'Support Agent') "
        f"GROUP BY {col_str} "
        f"HAVING total_tickets > 10 "
        f"ORDER BY last_opened DESC, total_tickets ASC"
    )

