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
