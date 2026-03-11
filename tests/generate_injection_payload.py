import pandas as pd
import os
import re

def extract_fragment_from_sql(sql):
    """
    尝试从完整 SQL 中提取注入 Payload 片段。
    策略:
    1. 如果包含 WHERE，截取 WHERE 之后的内容 (通常主要是 condition + payload)
    2. 如果包含 UNION，截取 UNION 及其之后的内容
    3. 如果很短 (比如 < 50 字符)，可能本身就是 payload，直接保留
    """
    sql_upper = sql.upper()
    
    # 策略 1: 截取 WHERE 后的部分
    if ' WHERE ' in sql_upper:
        parts = re.split(r'\sWHERE\s', sql, flags=re.IGNORECASE, maxsplit=1)
        if len(parts) > 1:
            fragment = parts[1].strip()
            # 进一步尝试去除开头的 "id=1" 等正常条件，但这比较复杂，
            # 简单策略: 如果片段里包含引号或注释符，更有可能是 payload
            return fragment

    # 策略 2: 截取 UNION 后的部分 (及 UNION 本身)
    if ' UNION ' in sql_upper:
        match = re.search(r'\s(UNION.*)', sql, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    # 策略 3: 如果本身就很短，或者是注释开头，保留
    if len(sql) < 60:
        return sql
        
    return None # 过于复杂的完整语句，如果提取不出特征，宁缺毋滥

def extract_and_save_payloads(input_csv, output_file):
    """
    从 Superviz25-SQL 数据集中提取 Label=1 (注入攻击) 的 Payload，
    并保存到文本文件供查看。
    """
    if not os.path.exists(input_csv):
        print(f"错误: 找不到输入文件: {input_csv}")
        # 尝试在上级目录查找
        alt_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'superviz25', 'Superviz25-SQL.csv')
        if os.path.exists(alt_path):
            input_csv = alt_path
            print(f"已自动定位到文件: {input_csv}")
        else:
            return

    try:
        print(f"正在读取数据集: {input_csv} ...")
        # 尝试不同的编码读取，并设置 low_memory=False 防止 mixed types 警告
        try:
            df = pd.read_csv(input_csv, encoding='utf-8', low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(input_csv, encoding='latin1', low_memory=False)
        
        # 归一化列名以适应不同数据集格式
        cols = {c.lower(): c for c in df.columns}
        # 增加 full_query (Superviz25-SQL常用列名)
        query_col = cols.get('query') or cols.get('payload') or cols.get('sentence') or cols.get('full_query')
        label_col = cols.get('label') or cols.get('class') or cols.get('label')

        if not query_col or not label_col:
            print(f"错误: 无法识别 query 或 label 列。当前列名: {list(df.columns)}")
            return

        # 筛选 Label=1 的样本 (假设 1 代表注入攻击)
        # 注意：有些数据集可能是字符串 label，需要检查
        # 这里统一转成数字处理
        df[label_col] = pd.to_numeric(df[label_col], errors='coerce').fillna(0)
        attack_df = df[df[label_col] == 1]
        
        raw_queries = attack_df[query_col].dropna().astype(str).unique()
        
        extracted_payloads = []
        for q in raw_queries:
            fragment = extract_fragment_from_sql(q)
            if fragment and len(fragment) > 3:
                extracted_payloads.append(fragment)

        # 去重
        clean_payloads = sorted(list(set(extracted_payloads)))

        print(f"原始攻击语句 {len(raw_queries)} 条，提取到 {len(clean_payloads)} 条潜在 Fragment Payloads。")

        # 保存到文件
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            for p in clean_payloads:
                f.write(p + '\n')
        
        print(f"Payload列表已保存至: {output_file}")

    except Exception as e:
        print(f"处理过程中发生错误: {e}")

if __name__ == "__main__":
    # 配置路径
    # 假设 Superviz25-SQL.csv 在项目的 data/superviz25 目录下，或者你可以修改为绝对路径
    input_dataset_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'Superviz25-SQL', 'Superviz25-SQL.csv')
    
    # 输出文件位置
    output_payload_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'injection_payloads.txt')
    
    extract_and_save_payloads(input_dataset_path, output_payload_file)
