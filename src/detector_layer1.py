from sklearn.ensemble import IsolationForest
import joblib
import numpy as np


class Layer1Detector:
    def __init__(self, model_name="isolation_forest"):
        if model_name == "isolation_forest":
            from models.isolation_forest import IsolationForestModel
            self.model = IsolationForestModel()
        else:
            raise ValueError(f"未知模型: {model_name}")

    def train(self, embeddings):
        self.model.train(embeddings)

    def detect(self, embeddings):
        return self.model.detect(embeddings)

