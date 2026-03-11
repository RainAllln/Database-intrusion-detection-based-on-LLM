import sys
import os
# 将根目录添加到sys.path中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ..generate_data_v2 import _safe_val, SCHEMA
import random

def build_sql(table, verb, cols):
    cols = SCHEMA[table]  # 使用表的全部列
    if table == "orders":
        if verb == "INSERT":
            return _role4_orders_insert(table, cols)
        elif verb == "UPDATE":
            return _role4_orders_update(table, cols)
        elif verb == "DELETE":
            return _role4_orders_delete(table, cols)
        elif verb == "SELECT":
            return _role4_orders_select(table, cols)
    elif table == "inventory":
        if verb == "INSERT":
            return _role4_inventory_insert(table, cols)
        elif verb == "UPDATE":
            return _role4_inventory_update(table, cols)
        elif verb == "DELETE":
            return _role4_inventory_delete(table, cols)
        elif verb == "SELECT":
            return _role4_inventory_select(table, cols)
    elif table == "suppliers":
        if verb == "INSERT":
            return _role4_suppliers_insert(table, cols)
        elif verb == "UPDATE":
            return _role4_suppliers_update(table, cols)
        elif verb == "DELETE":
            return _role4_suppliers_delete(table, cols)
        elif verb == "SELECT":
            return _role4_suppliers_select(table, cols)
    return f"-- 未定义 {table} 的 {verb} 操作"

def _role4_orders_insert(table, cols):
    specific_cols = random.sample(['order_id', 'ship_date', 'shipping_method', 'status'], k=random.randint(2, 4))
    col_str = ", ".join(specific_cols)
    vals = ", ".join([f"'{table}_{c}_val_{random.randint(100, 999)}'" for c in specific_cols])
    return f"INSERT INTO {table} ({col_str}) VALUES ({vals})"

def _role4_orders_update(table, cols):
    set_clause = ", ".join([f"{c} = '{table}_{c}_updated_{random.randint(1, 100)}'" for c in cols])
    return (
        f"UPDATE {table} SET {set_clause}, ship_date = CURRENT_DATE "
        f"WHERE {table}.order_id IN (SELECT order_id FROM {table} WHERE shipping_method = 'STANDARD') "
        f"AND {table}.status = 'PROCESSING'"
    )

def _role4_orders_delete(table, cols):
    return (
        f"DELETE FROM {table} "
        f"WHERE {table}.order_id IN (SELECT order_id FROM {table} WHERE shipping_method = 'EXPRESS') "
        f"AND NOT EXISTS (SELECT 1 FROM inventory WHERE inventory.item_id = {table}.order_id)"
    )

def _role4_orders_select(table, cols):
    col_str = ", ".join([f"{table}.{c}" for c in ["order_id", "ship_date", "shipping_method", "status"]])
    return (
        f"SELECT {col_str}, inventory.stock_qty "
        f"FROM {table} "
        f"LEFT JOIN inventory ON {table}.order_id = inventory.item_id "
        f"WHERE {table}.status = 'PROCESSING' "
        f"AND {table}.shipping_method IN (SELECT DISTINCT shipping_method FROM {table} WHERE status = 'SHIPPED') "
        f"ORDER BY {table}.ship_date ASC"
    )

def _role4_inventory_insert(table, cols):
    specific_cols = random.sample(['item_id', 'stock_qty', 'warehouse_loc', 'restock_level'], k=4)
    col_str = ", ".join(specific_cols)
    vals = ", ".join([f"'{table}_{c}_val_{random.randint(100, 999)}'" for c in specific_cols])
    return f"INSERT INTO {table} ({col_str}) VALUES ({vals})"

def _role4_inventory_update(table, cols):
    set_clause = ", ".join([f"{c} = '{table}_{c}_updated_{random.randint(1, 100)}'" for c in cols])
    return (
        f"UPDATE {table} SET {set_clause}, last_updated = CURRENT_TIMESTAMP "
        f"WHERE {table}.stock_qty < {table}.restock_level "
        f"AND {table}.warehouse_loc = 'MAIN_WAREHOUSE'"
    )

def _role4_inventory_delete(table, cols):
    return (
        f"DELETE FROM {table} "
        f"WHERE {table}.item_id IN (SELECT item_id FROM {table} WHERE stock_qty = 0) "
        f"AND {table}.restock_level > 50"
    )

def _role4_inventory_select(table, cols):
    col_str = ", ".join([f"{table}.{c}" for c in cols])
    return (
        f"SELECT {col_str}, suppliers.sup_name, suppliers.region "
        f"FROM {table} "
        f"LEFT JOIN suppliers ON {table}.sup_id = suppliers.sup_id "
        f"WHERE {table}.stock_qty < {table}.restock_level "
        f"AND suppliers.rating > 3 "
        f"AND {table}.warehouse_loc = 'MAIN_WAREHOUSE' "
        f"ORDER BY {table}.stock_qty ASC, suppliers.region DESC"
    )

def _role4_suppliers_insert(table, cols):
    specific_cols = random.sample(['sup_id', 'sup_name', 'contact_info', 'region'], k=4)
    col_str = ", ".join(specific_cols)
    vals = ", ".join([f"'{table}_{c}_val_{random.randint(100, 999)}'" for c in specific_cols])
    return f"INSERT INTO {table} ({col_str}) VALUES ({vals})"

def _role4_suppliers_update(table, cols):
    set_clause = ", ".join([f"{c} = '{table}_{c}_updated_{random.randint(1, 100)}'" for c in cols])
    return (
        f"UPDATE {table} SET {set_clause}, contract_end_date = CURRENT_DATE + INTERVAL '1 year' "
        f"WHERE {table}.rating < 3 "
        f"AND {table}.supplier_status = 'ACTIVE'"
    )

def _role4_suppliers_delete(table, cols):
    return (
        f"DELETE FROM {table} "
        f"WHERE {table}.sup_id IN (SELECT sup_id FROM {table} WHERE contract_end_date < CURRENT_DATE) "
        f"AND {table}.rating <= 2"
    )

def _role4_suppliers_select(table, cols):
    col_str = ", ".join([f"{table}.{c}" for c in cols])
    return (
        f"SELECT {col_str}, COUNT(inventory.item_id) AS total_items, AVG(inventory.stock_qty) AS avg_stock "
        f"FROM {table} "
        f"LEFT JOIN inventory ON {table}.sup_id = inventory.sup_id "
        f"WHERE {table}.supplier_status = 'ACTIVE' "
        f"AND {table}.rating >= 4 "
        f"GROUP BY {col_str} "
        f"HAVING total_items > 5 "
        f"ORDER BY avg_stock DESC, {table}.rating ASC"
    )
