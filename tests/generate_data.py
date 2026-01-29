import os
import pandas as pd
import random
import glob

# 配置路径
INPUT_DATA_PATH = '../data/raw/sqlInjection'  # 假设原始 SQLI 数据集在这里，如果是 CSV/TXT 文件
OUTPUT_DIR = '../data/custom'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'custom_dataset.csv')

# 确保输出目录存在
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 定义角色映射
# 0: 财务 (Finance)
# 1: 人事 (HR)
# 2: 销售 (Sales)
# 3: 开发 (Dev)
ROLE_MAP = {
    0: 'Finance',
    1: 'HR',
    2: 'Sales',
    3: 'Dev'
}

# 1. 定义每个表属于哪个角色 (表级权限)
ROLE_TABLES = {
    0: ['finance', 'salary', 'budget', 'revenue'],
    1: ['employee', 'staff', 'hr', 'attendance'],
    2: ['customer', 'orders', 'sales', 'leads'],
    3: ['logs', 'config', 'dev', 'test_table']
}

# 反向映射：表 -> 归属的角色
TABLE_OWNER = {table: role for role, tables in ROLE_TABLES.items() for table in tables}
# 定义每个角色允许访问的表
ALL_TABLES = list(TABLE_OWNER.keys())

# 2. 定义详细的 Schema (列级权限)
# normal: 该表拥有者可以正常查询的列
# sensitive: 即使是该表拥有者，查询这些列也被视为异常/伪装 (Label 2)
SCHEMA_DEFINITION = {
    # --- Role 0: 财务 ---
    'finance': {
        # 正常查询的是 记录id，交易日期，金额，部门id，项目代码，类别
        'normal': ['record_id', 'trans_date', 'amount', 'dept_id', 'project_code', 'category'],
        # 隐私列是 管理员签名，隐藏资产，审计日志
        'sensitive': ['admin_signature', 'hidden_assets', 'audit_trace_log']
    },
    'salary': {
        # 正常查询的是 员工编号， 基本工资，奖金，税前扣除，净工资，发薪日期
        'normal': ['emp_id', 'basic_salary', 'bonus', 'tax_deduction', 'net_pay', 'pay_date'],
        # 隐私列是 个人银行卡号，身份证号，家庭信息
        'sensitive': ['personal_bank_card_no', 'id_card', 'family_info']
    },
    'budget': {
        # 正常查询的是 年份，季度，部门名称，批准金额，剩余金额，状态
        'normal': ['year', 'quarter', 'dept_name', 'approved_amt', 'remain_amt', 'status'],
        # 隐私列是 经理审批密钥，秘密储备基金
        'sensitive': ['manager_approval_key', 'secret_reserve_fund']
    },
    'revenue': {
        # 正常查询的是 来源，渠道，毛收入，净收入，币种，地区
        'normal': ['source', 'channel', 'gross_income', 'net_income', 'currency', 'region'],
        # 隐私列是 合作伙伴佣金率，加密令牌
        'sensitive': ['partner_commission_rate', 'encrypted_token']
    },

    # --- Role 1: HR ---
    'employee': {
        # 正常查询的是 员工姓名，员工编号，性别，职位，部门，入职日期，邮箱
        'normal': ['emp_name', 'emp_id', 'gender', 'position', 'department', 'entry_date', 'email'],
        # 隐私列是 密码哈希，社会保障号码，政治面貌
        'sensitive': ['password_hash', 'social_security_number', 'political_status']
    },
    'staff': {
        # 正常查询的是 员工代码，工作地点，办公电话，直属经理，职位级别
        'normal': ['staff_code', 'work_loc', 'office_phone', 'direct_manager', 'job_level', 'emergency_contact_mobile'],
        # 隐私列是 家庭住址
        'sensitive': ['home_address']
    },
    'hr': {
        # 正常查询的是 政策编号，文档名称，发布日期，面试评分，简历路径
        'normal': ['policy_id', 'doc_name', 'publish_date', 'interview_score', 'resume_path'],
        # 隐私列是 背景调查结果
        'sensitive': ['background_check_result']
    },
    'attendance': {
        # 正常查询的是 签到时间，签退时间，日期，请假类型，加班小时数
        'normal': ['check_in', 'check_out', 'date', 'leave_type', 'overtime_hours'],
        # 隐私列是 生物识别数据
        'sensitive': ['biometric_data']
    },

    # --- Role 2: 销售 ---
    'customer': {
        # 正常查询的是 客户名称，客户等级，最后购买日期，积分余额，公开联系电话
        'normal': ['cust_name', 'cust_level', 'last_purchase', 'points', 'public_phone'],
        # 隐私列是 客户身份证号，家庭地址，信用卡
        'sensitive': ['cust_id_card', 'home_address', 'credit_card']
    },
    'orders': {
        # 正常查询的是 订单编号，产品名称，数量，总价，状态，发货日期
        'normal': ['order_id', 'prod_name', 'qty', 'total_price', 'status', 'ship_date'],
        # 隐私列是 账单地址，支付信息
        'sensitive': ['payment_gateway_token', 'fraud_check_score']
    },
    'sales': {
        # 正常查询的是 销售代表ID，月度目标，已完成金额，区域代码
        'normal': ['rep_id', 'monthly_target', 'achieved', 'region_code'],
        # 隐私列无
        'sensitive': []
    },
    'leads': {
        # 正常查询的是 公司名称，联系人，行业，意向等级，来源网站
        'normal': ['company', 'contact', 'industry', 'intent', 'source_web'],
        # 隐私列是 私人手机号，CEO邮箱
        'sensitive': ['private_mobile', 'ceo_email']
    },

    # --- Role 3: 开发 ---
    'logs': {
        # 正常查询的是 跟踪ID，日志级别，消息内容，时间戳，服务名称，IP地址
        'normal': ['trace_id', 'level', 'msg', 'timestamp', 'service', 'ip_addr'],
        # 隐私列是 用户会话令牌，Cookie内容
        'sensitive': ['user_session_token', 'cookie_content']
    },
    'config': {
        # 正常查询的是 参数名称，参数值，环境，是否激活，版本号
        'normal': ['key', 'value', 'env', 'is_active', 'version'],
        # 隐私列是 加密密钥，访问令牌
        'sensitive': ['db_password', 'secret_key', 'aws_access_key']
    },
    'dev': {
        # 正常查询的是 功能名称，测试用户ID，部署ID，分支名称
        'normal': ['feature', 'test_uid', 'deploy_id', 'branch'],
        # 隐私列是 管理员凭证
        'sensitive': ['prod_admin_credential']
    },
    'test_table': {
        # 正常查询的是 测试ID，字段1，字段2，描述
        'normal': ['id', 'field1', 'field2', 'desc'],
        # 隐私列无
        'sensitive': []
    }
}

def load_sqli_payloads(path):
    """
    加载原始 SQL 注入数据集中的 Payload。
    """
    payloads = []


    files = glob.glob(os.path.join(path, '*.csv')) + glob.glob(os.path.join(path, '*.txt'))

    for f in files:
        try:
            if f.endswith('.csv'):
                df = pd.read_csv(f)
                # 假设有一列叫 'Sentence' 或 'Query' 或 'Payload'
                # 请根据实际列名修改
                if 'Sentence' in df.columns:
                    payloads.extend(df['Sentence'].astype(str).tolist())
                elif 'Query' in df.columns:
                    payloads.extend(df['Query'].astype(str).tolist())
            else:
                with open(f, 'r', encoding='utf-8', errors='ignore') as file:
                    lines = [line.strip() for line in file if line.strip()]
                    payloads.extend(lines)
        except Exception as e:
            print(f"Error reading file {f}: {e}")

    # 去重
    return list(set(payloads))


def generate_normal_query(role_id):
    """
     Label 0: 正常业务操作
     1. 访问允许的表
     2. 只查询 'normal' 列 (禁止 *)
     3. 动词限制：
        - HR (1): SELECT, INSERT, UPDATE, DELETE
        - 其他 (0,2,3): SELECT, INSERT, UPDATE (严禁 DELETE)
        - 所有人: 严禁 DROP
     """
    allowed_tables = ROLE_TABLES[role_id]
    table = random.choice(allowed_tables)
    cols_def = SCHEMA_DEFINITION.get(table, {'normal': ['id'], 'sensitive': []})

    # 随机选取 1-3 个普通列
    selected_cols = random.sample(cols_def['normal'], k=min(len(cols_def['normal']), random.randint(1, 3)))
    cols_str = ", ".join(selected_cols)

    # 决定操作类型
    # HR 可以 DELETE，其他人不行
    if role_id == 1:
        ops = ['SELECT', 'SELECT', 'SELECT', 'INSERT', 'UPDATE', 'DELETE']
    else:
        ops = ['SELECT', 'SELECT', 'SELECT', 'INSERT', 'UPDATE']

    op = random.choice(ops)

    if op == 'SELECT':
        # 增加一些 WHERE 子句 Variations
        templates = [
            f"SELECT {cols_str} FROM {table}",
            f"SELECT {cols_str} FROM {table} WHERE {selected_cols[0]} = 'val'",
            f"SELECT {cols_str} FROM {table} ORDER BY {selected_cols[-1]} DESC LIMIT 10",
            f"SELECT count({selected_cols[0]}) FROM {table}"
        ]
    elif op == 'INSERT':
        val_placeholders = ", ".join(["'test'" for _ in selected_cols])
        templates = [f"INSERT INTO {table} ({cols_str}) VALUES ({val_placeholders})"]
    elif op == 'UPDATE':
        templates = [f"UPDATE {table} SET {selected_cols[0]}='new' WHERE {selected_cols[-1]}=1"]
    elif op == 'DELETE':
        templates = [f"DELETE FROM {table} WHERE {selected_cols[0]} = 'old_data'"]

    return random.choice(templates)


def generate_violation_query(role_id):
    """
    生成越权或伪装查询 (Label 2)
    包含以下几种情况：
    1. 访问了该角色无权访问的表 (越权)
    2. 访问了允许的表，但查询了敏感/隐私字段 (伪装/越权)
    3. 使用了 'SELECT *' (不符合只查特定列的规范，视为伪装)
    4. 使用了禁止的动词 (如非HR使用DELETE，或任何人使用DROP)
    """

    # 定义四种违规类型的权重/概率
    violation_type = random.choices(
        ['table_violation', 'column_violation', 'wildcard_violation', 'verb_violation'],
        weights=[0.3, 0.3, 0.2, 0.2],
        k=1
    )[0]

    allowed_tables = ROLE_TABLES[role_id]

    # --- Case 1: 表级越权 (访问不允许的表) ---
    if violation_type == 'table_violation':
        forbidden_tables = [t for t in ALL_TABLES if t not in allowed_tables]
        if not forbidden_tables:
            # 如果拥有所有权限（极少见），则回落到 wildcard
            violation_type = 'wildcard_violation'
        else:
            target_table = random.choice(forbidden_tables)
            # 即使查普通列，只要表不对，就是越权
            cols_def = SCHEMA_DEFINITION.get(target_table, {'normal': ['id']})
            col = random.choice(cols_def['normal'])
            return f"SELECT {col} FROM {target_table} LIMIT 1"

    # --- Case 2: 列级越权 (查了允许表中的敏感列) ---
    if violation_type == 'column_violation':
        # 寻找该角色名下含有敏感列的表
        tables_with_sensitive = [t for t in allowed_tables if SCHEMA_DEFINITION.get(t, {}).get('sensitive')]

        if tables_with_sensitive:
            target_table = random.choice(tables_with_sensitive)
            sensitive_col = random.choice(SCHEMA_DEFINITION[target_table]['sensitive'])

            templates = [
                f"SELECT {sensitive_col} FROM {target_table}",
                f"SELECT {sensitive_col}, id FROM {target_table} WHERE id=1",
                f"SELECT count({sensitive_col}) FROM {target_table}"  # 试图探测敏感数据分布
            ]
            return random.choice(templates)
        else:
            # 如果该角色表里没有敏感列，回落到 wildcard
            violation_type = 'wildcard_violation'

    # --- Case 3: 行为伪装 (使用 SELECT *) ---
    if violation_type == 'wildcard_violation':
        target_table = random.choice(allowed_tables)
        templates = [
            f"SELECT * FROM {target_table}",
            f"SELECT * FROM {target_table} WHERE id > 100",
            f"SELECT * FROM {target_table}"
        ]
        return random.choice(templates)

    # --- Case 4: 危险动作 (DROP / 非法DELETE) ---
    if violation_type == 'verb_violation':
        target_table = random.choice(allowed_tables)

        # 所有人都不允许 DROP
        drop_queries = [
            f"DROP TABLE {target_table}",
            f"TRUNCATE TABLE {target_table}"
        ]

        # 除非是HR (Role 1), 否则其他人不允许 DELETE
        delete_queries = [
            f"DELETE FROM {target_table}",
            f"DELETE FROM {target_table} WHERE id=1"
        ]

        if role_id != 1:  # 非 HR
            pool = drop_queries + delete_queries
        else:
            pool = drop_queries  # HR 只能触发 DROP 这种违规

        return random.choice(pool)

    return None


def main():
    print("开始生成数据集...")
    print(f"读取原始 SQL注入 Payloads: {INPUT_DATA_PATH}")
    sqli_payloads = load_sqli_payloads(INPUT_DATA_PATH)

    # 如果没有找到文件，使用一些模拟数据防止报错
    if not sqli_payloads:
        print("在指定路径未找到数据，生成少量模拟注入数据以供测试。")
        sqli_payloads = ["' OR 1=1 --", "admin' --", "UNION SELECT 1,2,3,4,5"]

    dataset = []

    # === 1. 生成 Label 1 (SQL注入攻击) ===
    # 逻辑：Payload本身就是攻击，角色是谁不重要（可能是被动触发或内部恶意人员）
    print(f"1. 生成 Label 1 (SQL注入) 样本: {len(sqli_payloads)} 条")
    for payload in sqli_payloads:
        role = random.randint(0, 3)
        dataset.append({
            'query': payload,
            'role': role,
            'Label': 1
        })

    # 确定基准数量，保持类别相对平衡
    # 假设我们需要 1:1:1 的比例，或者是根据 payload 数量来定
    n_samples = len(sqli_payloads)

    # 如果 payload 太多，可以适当限制 n_samples 上限，比如 10000，避免生成太慢
    # n_samples = min(n_samples, 10000)

    # === 2. 生成 Label 0 (正常访问) ===
    # 逻辑：动词合规、表合规、列合规（只查normal列）
    print(f"2. 生成 Label 0 (正常访问) 样本: {n_samples} 条")
    for _ in range(n_samples):
        role = random.randint(0, 3)
        query = generate_normal_query(role)
        dataset.append({
            'query': query,
            'role': role,
            'Label': 0
        })

    # === 3. 生成 Label 2 (越权/伪装/违规) ===
    # 逻辑：覆盖表越权、敏感列越权、*号伪装、危险动词
    print(f"3. 生成 Label 2 (越权/伪装) 样本: {n_samples} 条")
    for _ in range(n_samples):
        role = random.randint(0, 3)
        query = generate_violation_query(role)
        if query:
            dataset.append({
                'query': query,
                'role': role,
                'Label': 2
            })

    # 转换为 DataFrame 并打乱顺序
    df = pd.DataFrame(dataset)
    df = df.sample(frac=1).reset_index(drop=True)

    # 简单的数据清洗，去除空值
    df.dropna(subset=['query'], inplace=True)

    # 打印一些示例进行检查
    print("\n生成的样本示例:")
    print(df.groupby('Label').apply(lambda x: x.sample(2) if len(x) >= 2 else x))

    # 保存
    print(f"\n保存数据集到 {OUTPUT_FILE}，共 {len(df)} 行。")
    try:
        df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
        print("文件保存成功！")
    except Exception as e:
        print(f"保存文件失败: {e}")


if __name__ == "__main__":
    main()