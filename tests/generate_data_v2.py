import os
import pandas as pd
import random
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

import importlib  # 用于动态导入模块

# --- 1. 完整表结构定义 (Schema) ---
# 根据 README 合并了所有列，取消了敏感列的区分
SCHEMA = {
    'finance': ['record_id', 'trans_date', 'amount', 'dept_id', 'project_code', 'category','admin_signature', 'hidden_assets', 'audit_trace_log'],
    'salary': ['salary_id','emp_id', 'basic_salary', 'bonus', 'tax_deduction', 'net_pay', 'pay_date','personal_bank_card_no', 'id_card', 'family_info'],
    'budget': ['budget_id', 'year', 'quarter', 'dept_name', 'approved_amt', 'remain_amt', 'status','manager_approval_key', 'secret_reserve_fund'],
    'revenue': ['rep_id', 'month', 'region', 'gross_revenue', 'net_profit','source', 'channel', 'gross_income', 'net_income', 'currency', 'region','partner_commission_rate', 'encrypted_token'],
    'employee': ['emp_name', 'emp_id', 'gender', 'position', 'department', 'entry_date', 'email', 'rank', 'status', 'political_status', 'phone_number'],
    'staff': ['staff_id', 'staff_code', 'work_loc', 'office_phone', 'direct_manager', 'job_level', 'emergency_contact_mobile', 'home_address','work_email','national_id'],
    'hr': ['policy_id', 'doc_name', 'publish_date', 'interview_score', 'resume_path','background_check_result', 'interview_notes', 'status', 'work_experience'],
    'attendance': ['attendance_id','check_in', 'check_out', 'date', 'leave_type', 'overtime_hours', 'biometric_data', 'shift_id', 'leave_balance', 'attendance_status'],
    'customer': ['customer_id','cust_name', 'cust_level', 'last_purchase', 'points', 'public_phone', 'cust_id_card', 'home_address', 'credit_card', 'email', 'account_status'],
    'orders': ['order_id', 'prod_name', 'qty', 'total_price', 'status', 'ship_date', 'payment_gateway_token', 'fraud_check_score', 'discount', 'shipping_method'],
    'sales': ['sales_id', 'rep_id', 'monthly_target', 'achieved', 'region_code', 'lead_count', 'target_sales_amount', 'territory', 'customer_feedback', 'total_sales_value'],
    'leads': ['leads_id', 'company', 'contact', 'industry', 'intent', 'source_web', 'private_mobile', 'ceo_email', 'lead_status', 'lead_source', 'email_opt_in'],
    'logs': ['log_id', 'level', 'msg', 'timestamp', 'service', 'ip_addr', 'user_session_token', 'cookie_content', 'request_method', 'user_agent'],
    'config': ['config_id', 'key', 'value', 'env', 'is_active', 'version', 'db_password', 'secret_key', 'aws_access_key', 'config_type', 'last_modified'],
    'dev': ['dev_id', 'feature', 'test_uid', 'deploy_id', 'branch' , 'prod_admin_credential', 'commit_hash', 'test_environment', 'deployment_date', 'version'],
    'test_table': ['test_id', 'field1', 'field2', 'desc', 'created_at', 'updated_at', 'test_status', 'test_result', 'error_message', 'test_duration'],
    'inventory': ['inventory_id', 'item_id', 'sku', 'stock_qty', 'warehouse_loc', 'sup_id', 'purchase_price', 'sale_price', 'supplier_name', 'restock_level', 'item_description'],
    'suppliers': ['sup_id', 'sup_name', 'contact_info', 'category', 'region', 'rating', 'contract_start_date', 'contract_end_date', 'supplier_status', 'payment_terms'],
    'campaigns': ['camp_id', 'camp_name', 'budget_limit', 'channel', 'start_date', 'end_date', 'target_audience', 'actual_spend', 'conversion_rate', 'roi'],
    'tickets': ['ticket_id', 'issue_type', 'status', 'user_id', 'priority', 'assigned_to', 'ticket_opened_date', 'ticket_closed_date', 'resolution_notes', 'feedback'],
    'performance': ['perfomance_id', 'rev_id', 'emp_id', 'rating', 'feedback', 'review_date', 'reviewer_id', 'performance_category', 'goals_achieved', 'training_completed', 'promotion_eligible']
}


ALL_TABLES = list(SCHEMA.keys())

# --- 2. 角色与权限映射 (基于 README.md 重叠表) ---
# W = 可执行 SELECT, INSERT, UPDATE, DELETE
# R = 仅执行 SELECT
# 角色0：财务
# 角色1：人事
# 角色2：销售
# 角色3：系统运维
# 角色4：物流仓储
# 角色5：市场推广
# 角色6： 审计
# 角色7： 客户服务
ROLE_PERMISSIONS = {
    0: {'finance': 'W', 'salary': 'W', 'budget': 'W', 'revenue': 'W'},
    1: {'employee': 'W', 'staff': 'W', 'hr': 'W', 'attendance': 'W', 'performance': 'W'},
    2: {'customer': 'W', 'orders': 'W', 'sales': 'W', 'leads': 'W'},
    3: {'dev': 'W', 'logs': 'W', 'config': 'W', 'test_table': 'W'},
    4: {'inventory': 'W', 'suppliers': 'W', 'orders': 'W'},
    5: {'campaigns': 'W', 'customer': 'W', 'revenue': 'W'},
    6: {t: 'R' for t in ALL_TABLES},  # Audit (Role 6) 拥有全局 R 权限
    7: {'customer': 'W', 'tickets': 'W', 'orders': 'W'}
}



# --- 新增: 加载外部数据集 Payloads ---
EXTERNAL_PAYLOADS = []

def load_external_payloads(filepath):
    """
    从 text 文件加载真实 Payload 片段 (一行一个)
    """
    global EXTERNAL_PAYLOADS
    if not os.path.exists(filepath):
        print(f"[Warn] 外部Payload文件未找到: {filepath}，将仅使用内置合成规则。")
        return

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            # 读取非空行
            lines = [line.strip() for line in f if line.strip()]
        
        # 简单过滤长度，避免过长的完整语句导致拼接异常
        EXTERNAL_PAYLOADS = [p for p in lines if len(p) < 150]
        
        print(f"[Info] 已加载外部 Payload 片段 {len(EXTERNAL_PAYLOADS)} 条，来源: {filepath}")

    except Exception as e:
        print(f"[Error] 加载外部Payload文件失败: {e}")


def _pick_cols(table, k_min=6, k_max=7):
    """从表的列中随机选择 k_min 到 k_max 列"""
    return random.sample(SCHEMA[table], k=random.randint(k_min, min(k_max, len(SCHEMA[table]))))


def build_simple_select(table, cols):
    """为 Role 6 生成简单的、无嵌套/UNION 的查询"""
    col_str = ", ".join([f"{table}.{c}" for c in cols])
    # 简单的 WHERE 和 ORDER BY，增加长度但不增加结构复杂度
    where_clause = f"{table}.{cols[0]} IS NOT NULL AND {table}.{random.choice(SCHEMA[table])} != 'deleted'"
    return f"SELECT {col_str} FROM {table} WHERE {where_clause} ORDER BY {table}.{cols[0]} ASC"

def get_diverse_injection(table, cols):
    """强制使用外部数据集导入的 Payload 片段生成注入语句 (支持多列)"""
    if not EXTERNAL_PAYLOADS:
        return f"SELECT * FROM {table} WHERE {cols[0]} = 'EXTERNAL_PAYLOAD_MISSING'"
    
    raw_fragment = random.choice(EXTERNAL_PAYLOADS)
    col_primary = cols[0]
    col_str = ", ".join(cols)
    
    # 构造注入语句结构
    template_type = random.choice(['where_eq', 'where_like', 'union_string', 'insert_inject'])
    
    if template_type == 'where_eq':
        return f"SELECT * FROM {table} WHERE {col_primary} = '1' /* {raw_fragment} */"
    elif template_type == 'where_like':
        return f"SELECT * FROM {table} WHERE {col_primary} LIKE '%a%' /* {raw_fragment} */"
    elif template_type == 'insert_inject':
        # 为多列生成对应的占位值
        other_vals = ", ".join([f"'{random.randint(1, 100)}'" for _ in range(len(cols)-1)])
        vals_str = f"'1' /* {raw_fragment} */"
        if other_vals:
            vals_str += f", {other_vals}"
        return f"INSERT INTO {table} ({col_str}) VALUES ({vals_str})"
    else:
        return f"SELECT {col_str} FROM {table} WHERE id=1 /* {raw_fragment} */"

def build_complex_select(table, cols):
    """生成带有高复杂度和长度的 SELECT 查询 (增加 JOIN, 复杂 WHERE, 聚合)"""
    q_type = random.choice(['simple_long', 'union', 'nested_join'])
    col_str = ", ".join([f"{table}.{c}" for c in cols])
    
    # 构建复杂的 WHERE 子句以增加长度
    extra_conditions = [
        f"{table}.{random.choice(SCHEMA[table])} IS NOT NULL",
        f"{table}.{random.choice(SCHEMA[table])} <> 'temporary'",
        f"{table}.{cols[0]} BETWEEN 10 AND 5000",
        f"{table}.{random.choice(SCHEMA[table])} LIKE '%config%'"
    ]
    where_clause = " AND ".join(random.sample(extra_conditions, k=random.randint(2, 3)))

    if q_type == 'simple_long':
        order_col = random.choice(SCHEMA[table])
        return f"SELECT {col_str} FROM {table} WHERE {where_clause} GROUP BY {col_str} ORDER BY {table}.{order_col} DESC"
    
    elif q_type == 'union':
        other_where = f"{table}.{cols[0]} = 'ARCHIVED' OR {table}.{cols[0]} = 'ACTIVE'"
        return f"SELECT {col_str} FROM {table} WHERE {where_clause} UNION ALL SELECT {col_str} FROM {table} WHERE {other_where}"
    
    else:
        # 增加 JOIN 操作以显著增加长度
        join_table = random.choice(ALL_TABLES)
        join_col = random.choice(SCHEMA[join_table])
        return (f"SELECT {col_str}, {join_table}.{join_col} FROM {table} "
                f"LEFT JOIN {join_table} ON {table}.{cols[0]} = {join_table}.{join_col} "
                f"WHERE {where_clause} AND {join_table}.{join_col} IS NOT NULL "
                f"AND {table}.{cols[-1]} IN (SELECT {cols[-1]} FROM {table} WHERE {cols[0]} > 0)")


def build_dml(verb, table, cols):
    """生成合规但较长的写操作 SQL"""
    col_str = ", ".join(cols)
    if verb == 'INSERT':
        # 插入更多伪数据以增加长度，匹配传入的 3-4 列
        vals = ", ".join([f"'{c}_val_{random.randint(100,999)}'" for c in cols])
        return f"INSERT INTO {table} ({col_str}, created_at, updated_by) VALUES ({vals}, CURRENT_TIMESTAMP, 'SYSTEM_ADMIN')"
    elif verb == 'UPDATE':
        set_clause = ", ".join([f"{c} = 'updated_{random.randint(1,100)}'" for c in cols])
        return f"UPDATE {table} SET {set_clause} WHERE {cols[0]} = 'target_id' AND status = 'VALID' AND version > 0"
    elif verb == 'DELETE':
        return f"DELETE FROM {table} WHERE {cols[0]} IN (SELECT {cols[0]} FROM {table} WHERE status = 'EXPIRED') AND permanence = 1"

def _safe_val(value):
    """
    确保值是安全的 SQL 值，避免注入风险。
    如果值是字符串，则加上引号；如果是数字，则直接返回。
    """
    # 转义单引号
    if isinstance(value, str):
        # 修复嵌套单引号问题
        escaped_value = value.replace("'", "''")  # 转义单引号
        return f"'{escaped_value}'"
    return str(value)

ROLE_TEMPLATES = {
    0: "role0",
    1: "role1",
    2: "role2",
    3: "role3",
    4: "role4",
    5: "role5",
    6: "role6",
    7: "role7",
}

def build_role_sql(role, table, verb, cols):
    """根据角色调用专属模板生成 SQL"""
    role_module = ROLE_TEMPLATES.get(role)
    if not role_module:
        return f"-- 未定义角色 {role} 的模板"
    try:
        # 修复动态导入路径，确保从当前脚本目录加载模块
        role_functions = importlib.import_module(f"tests.generate_data_v2_role.{role_module}")
        return role_functions.build_sql(table, verb, cols)
    except ImportError as e:
        # 输出更详细的错误信息
        return f"-- 无法加载角色 {role} 的模板模块: {e}"

def generate_query(role, label):
    perms = ROLE_PERMISSIONS[role]
    allowed_tables = list(perms.keys())

    if label == 0:
        # 正常操作 (Label 0)
        table = random.choice(allowed_tables)
        perm = perms[table]
        verb = random.choice(['SELECT', 'INSERT', 'UPDATE', 'DELETE']) if perm == 'W' else 'SELECT'
        cols = _pick_cols(table, k_min=6, k_max=7)
        return build_role_sql(role, table, verb, cols)

    elif label == 1:
        # 注入攻击 (Label 1)
        table = random.choice(allowed_tables) if allowed_tables else random.choice(ALL_TABLES)
        n_cols = random.randint(3, 4)
        cols = random.sample(SCHEMA[table], k=min(n_cols, len(SCHEMA[table])))
        return get_diverse_injection(table, cols)

    elif label == 2:
        # 伪装/内部攻击 (Label 2)
        violation_type = random.choice(['drop_truncate', 'unauthorized_access', 'write_without_permission'])

        if violation_type == 'drop_truncate':
            verb = random.choice(['DROP TABLE', 'TRUNCATE TABLE'])
            table = random.choice(ALL_TABLES)
            return f"{verb} {table}"

        elif violation_type == 'write_without_permission':
            if role == 6:  # Audit 尝试写
                table = random.choice(ALL_TABLES)
                verb = random.choice(['INSERT', 'UPDATE', 'DELETE'])
            else:
                unauthorized_tables = [t for t in ALL_TABLES if t not in allowed_tables]
                if unauthorized_tables:
                    table = random.choice(unauthorized_tables)
                else:
                    table = random.choice(allowed_tables)
                verb = random.choice(['INSERT', 'UPDATE', 'DELETE'])
            n_cols = random.randint(3, 4)
            cols = random.sample(SCHEMA[table], k=min(n_cols, len(SCHEMA[table])))
            return build_dml(verb, table, cols)

        else:
            unauthorized_tables = [t for t in ALL_TABLES if t not in allowed_tables]
            if unauthorized_tables:
                table = random.choice(unauthorized_tables)
            else:
                table = random.choice(ALL_TABLES)
            n_cols = random.randint(3, 4)
            cols = random.sample(SCHEMA[table], k=min(n_cols, len(SCHEMA[table])))
            verb = random.choice(['SELECT', 'INSERT', 'UPDATE', 'DELETE'])
            if verb == 'SELECT':
                if role == 6:
                    return build_simple_select(table, cols)
                return build_complex_select(table, cols)
            else:
                return build_dml(verb, table, cols)


def main(n_samples=150000):
    dataset = []
    print(f"开始生成规模为 {n_samples} 的高复杂度数据集...")
    
    # 移除预处理器引用，直接生成原始数据
    # 加载外部 Payload txt
    # 假设 generate_injection_payload.py 已经运行并生成了该文件
    payload_txt_path = '../data/processed/injection_payloads.txt'
    
    # 尝试自动定位文件 (处理相对路径可能的问题)
    if not os.path.exists(payload_txt_path):
        # 尝试相对于脚本的路径
        payload_txt_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'injection_payloads.txt')

    load_external_payloads(payload_txt_path) 

    dataset = []
    
    # 获取角色概率分布权重的列表 (0-7, 6是Audit)
    roles = list(range(8))
    
    for i in range(n_samples):
        lbl = random.choices([0, 1, 2], weights=[0.4, 0.3, 0.3])[0]
        role = random.randint(0, 7)  # Role 0 - 7

        raw_query = generate_query(role, lbl)

        dataset.append({
            'query': raw_query, 
            'role': role, 
            'Label': lbl
        })
        
        if (i + 1) % 10000 == 0:
            print(f"进度: {i + 1}/{n_samples}")

    df = pd.DataFrame(dataset)
    # 打乱数据
    df = df.sample(frac=1).reset_index(drop=True)

    # 确保存储目录存在
    os.makedirs('../data/custom', exist_ok=True)
    output_path = '../data/custom/complex_dataset_v3.csv'
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"数据集生成完毕！已保存至: {output_path}")
    print("\n生成的样本分布预览:")
    print(df['Label'].value_counts())


if __name__ == "__main__":
    main()

