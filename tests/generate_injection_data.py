import os
import pandas as pd
import random
import re

# 1. 配置路径 (抄自 generate_data.py)
INPUT_FILE = '../data/custom/custom_dataset.csv'
OUTPUT_DIR = '../data/custom'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'custom_dataset2.csv')

# 2. 定义角色和 Schema (与你的逻辑保持一致)
ROLE_TABLES = {
    0: ['finance', 'salary', 'budget', 'revenue'],
    1: ['employee', 'staff', 'hr', 'attendance'],
    2: ['customer', 'orders', 'sales', 'leads'],
    3: ['logs', 'config', 'dev', 'test_table']
}

SCHEMA_DEFINITION = {
    'finance': {'normal': ['dept_id', 'currency'], 'sensitive': ['profit', 'tax_id']},
    'salary': {'normal': ['emp_id', 'basic_salary'], 'sensitive': ['bonus', 'personal_bank_card_no']},
    'employee': {'normal': ['name', 'entry_date'], 'sensitive': ['phone', 'id_card']},
    'customer': {'normal': ['cust_name', 'last_purchase'], 'sensitive': ['email', 'address']},
    'logs': {'normal': ['timestamp', 'level'], 'sensitive': ['ip_address', 'cookie_content']},
    # 其他表省略，脚本会动态处理
}


def get_role_context(role_id):
    """为指定角色获取一个随机表名和该表的随机列名"""
    table = random.choice(ROLE_TABLES[role_id])
    # 如果定义里没有，默认给一些通用列名
    cols = SCHEMA_DEFINITION.get(table, {'normal': ['id', 'name']})['normal']
    col = random.choice(cols)
    return table, col


def transform_to_role_based_injection(original_payload, role_id):
    """
    核心逻辑：将原始攻击 Payload 包装进该角色的业务 SQL 结构中
    """
    table, col = get_role_context(role_id)

    # 清理原始 payload 里的多余引号，防止 SQL 结构过乱
    clean_payload = str(original_payload).strip().replace('"', "'")

    # 模拟几种常见的攻击注入点，模仿 Superviz25 的高级风格
    templates = [
        # 1. WHERE 注入 (最常见)
        f"SELECT {col} FROM {table} WHERE {col} = 'val' AND ({clean_payload})",
        # 2. UNION 注入 (带业务背景)
        f"SELECT {col} FROM {table} UNION {clean_payload}",
        # 3. 报错注入/堆叠注入
        f"UPDATE {table} SET {col} = 'test' WHERE id = 1; {clean_payload}",
        # 4. 原始风格保留 (部分保留，防止模型只学固定模板)
        f"{table} {clean_payload}"
    ]

    return random.choice(templates)


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"错误: 找不到输入文件 {INPUT_FILE}")
        return

    print(f"正在读取旧数据集: {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)

    print("正在进行 Label 1 (注入攻击) 角色化改造...")

    # 只针对 Label 为 1 的行进行处理
    def process_row(row):
        if row['Label'] == 1:
            # 修改 query，带上角色的表名
            return transform_to_role_based_injection(row['query'], row['role'])
        return row['query']

    df['query'] = df.apply(process_row, axis=1)

    # 确保输出目录存在
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print(f"正在保存新数据集到: {OUTPUT_FILE}...")
    df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
    print("处理完成！")


if __name__ == "__main__":
    main()