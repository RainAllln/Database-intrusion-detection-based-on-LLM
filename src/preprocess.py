import re
import sqlparse

class SQLPreprocessor:
    def __init__(self):
        # 定义替换逻辑
        self.num_pattern = re.compile(r'\b\d+\b')
        self.str_pattern = re.compile(r"'.*?'")

    def normalize(self, sql):
        """将具体值转换为占位符，重点解决缺陷检测粒度问题 """
        if not isinstance(sql, str): return ""
        sql = sql.lower()
        sql = self.num_pattern.sub('<NUM>', sql)
        sql = self.str_pattern.sub('<STR>', sql)
        # 去除多余空格
        sql = " ".join(sql.split())
        return sql

    def get_ast_sequence(self, sql):
        """将SQL解析为token序列（AST展平）"""
        if not isinstance(sql, str): return []
        parsed = sqlparse.parse(sql)
        if not parsed:
            return []
        tokens = []

        def flatten(token_list):
            for token in token_list:
                if token.is_group:
                    yield from flatten(token.tokens)
                else:
                    # 只保留有意义的token
                    val = token.value.strip()
                    if val:
                        yield val.lower()

        for stmt in parsed:
            tokens.extend(flatten(stmt.tokens))
        return tokens
