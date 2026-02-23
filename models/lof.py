from sklearn.neighbors import LocalOutlierFactor

class LOFModel:
    def __init__(self, n_neighbors=20, contamination=0.05):
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.model = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            contamination=self.contamination,
            novelty=True,
            n_jobs=-1
        )

    def train(self, embeddings):
        self.model.fit(embeddings)

    def detect(self, embeddings):
        scores = self.model.decision_function(embeddings)
        preds = self.model.predict(embeddings)
        return preds, scores

    def get_hyperparams(self):
        """返回当前 LOF 的核心超参数."""
        return {
            "n_neighbors": self.n_neighbors,
            "contamination": self.contamination,
            "novelty": self.model.novelty,
            "n_jobs": self.model.n_jobs,
        }

    def get_hyperparams_str(self) -> str:
        """
        返回格式化后的超参数字符串，供 main.py 直接写入 model_paras。
        """
        h = self.get_hyperparams()
        lines = [
            "Layer1: LOF (trained on normal samples)",
            f"  - n_neighbors: {h.get('n_neighbors')}",
            f"  - contamination: {h.get('contamination')}",
            f"  - novelty: {h.get('novelty')}",
            f"  - n_jobs: {h.get('n_jobs')}",
        ]
        return "\n".join(lines)
