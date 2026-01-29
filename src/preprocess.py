import re

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
        """扩展：如果未来你想尝试AST展平序列，可以在此实现 """
        # 预留给未来择优方案
        pass