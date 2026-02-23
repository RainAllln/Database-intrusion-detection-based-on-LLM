from sklearn.ensemble import IsolationForest
import joblib
import numpy as np


class Layer1Detector:
    def __init__(self, model_name="isolation_forest"):
        if model_name == "isolation_forest":
            from models.isolation_forest import IsolationForestModel
            self.model = IsolationForestModel()
        elif model_name == "one_class_svm":
            from models.one_class_svm import OneClassSVMModel
            self.model = OneClassSVMModel()
        elif model_name == "lof":
            from models.lof import LOFModel
            self.model = LOFModel()
        else:
            raise ValueError(f"未知模型: {model_name}")

    def train(self, embeddings):
        self.model.train(embeddings)

    def detect(self, embeddings):
        return self.model.detect(embeddings)

    def get_hyperparams(self):
        """返回底层模型的超参数（如果支持）."""
        if hasattr(self.model, "get_hyperparams"):
            return self.model.get_hyperparams()
        return {}

    def get_hyperparams_str(self) -> str:
        """
        返回底层模型生成的超参数字符串，如果不存在则返回空字符串。
        以后更换第一层模型，只要底层实现 get_hyperparams_str 即可。
        """
        if hasattr(self.model, "get_hyperparams_str"):
            return self.model.get_hyperparams_str()
        return "Layer1: (no hyperparams_str defined)"

