from sklearn.ensemble import IsolationForest

class IsolationForestModel:
    def __init__(self):
        self.model = IsolationForest(
            n_estimators=200,
            contamination=0.05,
            max_samples='auto',
            random_state=42,
            n_jobs=-1
        )

    def train(self, embeddings):
        self.model.fit(embeddings)

    def detect(self, embeddings):
        scores = self.model.decision_function(embeddings)
        preds = self.model.predict(embeddings)
        return preds, scores

    def get_hyperparams(self):
        """返回当前IsolationForest的核心超参数，用于记录实验."""
        return {
            "n_estimators": self.model.n_estimators,
            "max_samples": self.model.max_samples,
            "contamination": self.model.contamination,
            "random_state": self.model.random_state,
            "n_jobs": self.model.n_jobs,
            "bootstrap": getattr(self.model, "bootstrap", False),
        }

    def get_hyperparams_str(self) -> str:
        """
        返回格式化后的超参数字符串，供 main.py 直接写入 model_paras。
        以后如果更换为别的第一层模型，只需在对应模型实现同名方法即可。
        """
        h = self.get_hyperparams()
        lines = [
            "Layer1: Isolation Forest (trained on normal samples)",
            f"  - n_estimators: {h.get('n_estimators')}",
            f"  - max_samples: {h.get('max_samples')}",
            f"  - contamination: {h.get('contamination')}",
            f"  - random_state: {h.get('random_state')}",
            f"  - n_jobs: {h.get('n_jobs')}",
            f"  - bootstrap: {h.get('bootstrap')}",
        ]
        return "\n".join(lines)
