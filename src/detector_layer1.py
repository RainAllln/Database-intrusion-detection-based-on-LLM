from sklearn.ensemble import IsolationForest
import joblib
import numpy as np


class Layer1Detector:
    def __init__(self, contamination=0.05):
        # 孤立森林：增加树的数量(n_estimators)提高稳定性，n_jobs=-1启用并行计算
        self.model = IsolationForest(
            n_estimators=200, 
            contamination=contamination, 
            max_samples='auto',
            random_state=42,
            n_jobs=-1
        )

    def train(self, embeddings):
        """仅使用正常SQL进行训练，识别离群点 """
        self.model.fit(embeddings)

    def detect(self, embeddings):
        """判定是否为离群点（-1为异常，1为正常） """
        scores = self.model.decision_function(embeddings)
        preds = self.model.predict(embeddings)
        return preds, scores
