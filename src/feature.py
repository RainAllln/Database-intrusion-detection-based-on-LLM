class SQLEmbedder:
    """SQL 特征提取统一接口，内部根据 extractor_type 选择具体 LLM 模型"""
    def __init__(self, extractor_type: str = "distilbert", **kwargs):
        self.extractor = self._load_extractor(extractor_type, **kwargs)

    def _load_extractor(self, extractor_type: str, **kwargs):
        """根据类型加载对应的特征提取器"""
        if extractor_type == "distilbert":
            from models.distilbert import DistilBERTFeatureExtractor
            return DistilBERTFeatureExtractor(**kwargs)
        elif extractor_type == "bert":
            from models.bert import BERTFeatureExtractor
            return BERTFeatureExtractor(**kwargs)
        elif extractor_type == "roberta":
            from models.roberta import RoBERTaFeatureExtractor
            return RoBERTaFeatureExtractor(**kwargs)
        elif extractor_type == "codebert":
            from models.codebert import CodeBERTFeatureExtractor
            return CodeBERTFeatureExtractor(**kwargs)
        else:
            raise ValueError(f"未知的特征提取器类型: {extractor_type}")

    def get_embeddings(self, texts, batch_size: int = 16):
        return self.extractor.get_embeddings(texts, batch_size=batch_size)

