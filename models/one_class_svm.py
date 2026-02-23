from sklearn.svm import OneClassSVM

class OneClassSVMModel:
    def __init__(self, kernel="rbf", gamma="scale", nu=0.05):
        self.model = OneClassSVM(kernel=kernel, gamma=gamma, nu=nu)

    def train(self, embeddings):
        self.model.fit(embeddings)

    def detect(self, embeddings):
        scores = self.model.decision_function(embeddings)
        preds = self.model.predict(embeddings)
        return preds, scores

    def get_hyperparams(self):
        """返回当前 One-Class SVM 的核心超参数，用于记录实验."""
        return {
            "kernel": self.model.kernel,
            "gamma": self.model.gamma,
            "nu": self.model.nu,
        }

    def get_hyperparams_str(self) -> str:
        """
        返回格式化后的超参数字符串，供 main.py 直接写入 model_paras。
        以后如果更换为别的第一层模型，只需在对应模型实现同名方法即可。
        """
        h = self.get_hyperparams()
        lines = [
            "Layer1: One-Class SVM (trained on normal samples)",
            f"  - kernel: {h.get('kernel')}",
            f"  - gamma: {h.get('gamma')}",
            f"  - nu: {h.get('nu')}",
        ]
        return "\n".join(lines)
