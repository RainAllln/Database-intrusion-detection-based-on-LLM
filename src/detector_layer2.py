import numpy as np
import torch

class Layer2Classifier:
    """
    第二层模型调度器：根据 model_name 选择不同实现，
    并统一 train / predict / get_hyperparams_str 接口。
    """
    def __init__(self, model_name: str = "mlp", input_dim: int = 772, num_roles: int = 4, device=None):
        self.model_name = model_name
        self.input_dim = input_dim
        self.num_roles = num_roles
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model_name == "mlp":
            from models.mlp import MLPClassifierModel
            self.model = MLPClassifierModel(
                input_dim=self.input_dim,
                num_roles=self.num_roles,
                device=self.device,
            )
        elif model_name == "lightgbm":
            # LightGBM 树模型（CPU 模型，device 仅为保持接口一致）
            from models.lightgbm import LightGBMClassifierModel
            self.model = LightGBMClassifierModel(
                input_dim=self.input_dim,
                num_roles=self.num_roles,
                device=self.device,
            )
        elif model_name == "distilbert_clf":
            # 使用 DistilBERT 自带分类头的分类器
            from models.distilbert import DistilBERTClassifierModel
            self.model = DistilBERTClassifierModel(
                num_roles=self.num_roles,
                device=self.device,
            )
        # 将来要扩展别的模型，只需要在这里加分支：
        # elif model_name == "lstm":
        #     from models.lstm_classifier import LSTMClassifierModel
        #     self.model = LSTMClassifierModel(...)
        else:
            raise ValueError(f"未知模型: {model_name}")

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              training_hparams: dict = None, output_dir: str = None):
        """
        统一训练入口，真正的训练逻辑由各个模型自己实现。
        返回 loss_history(list[float]) 或 None。
        """
        # 对于 distilbert_clf，我们期望 X_train 传入 dict:
        # {"texts": list[str], "rag_features": np.ndarray}
        if self.model_name == "distilbert_clf":
            if not isinstance(X_train, dict):
                raise ValueError("distilbert_clf 训练时，X_train 需为 {'texts', 'rag_features'} 字典")
            texts = X_train["texts"]
            rag_features = X_train["rag_features"]
            if hasattr(self.model, "train_model"):
                return self.model.train_model(
                    texts, rag_features, y_train,
                    training_hparams or {}, output_dir
                )
            raise NotImplementedError("distilbert_clf 未实现 train_model 方法")

        # 其他模型保持原来的行为：直接用向量特征
        if hasattr(self.model, "train_model"):
            return self.model.train_model(X_train, y_train, training_hparams or {}, output_dir)
        raise NotImplementedError(f"{self.model_name} 未实现 train_model 方法")

    def predict(self, X: np.ndarray):
        """
        统一预测接口。
        返回:
          y_pred: np.ndarray[int], shape (N,)
          probs:  np.ndarray[float], shape (N, num_roles)
        """
        # 对于 distilbert_clf，X 也为 dict:
        # {"texts": list[str], "rag_features": np.ndarray}
        if self.model_name == "distilbert_clf":
            if not isinstance(X, dict):
                raise ValueError("distilbert_clf 预测时，X 需为 {'texts', 'rag_features'} 字典")
            texts = X["texts"]
            rag_features = X["rag_features"]
            if hasattr(self.model, "predict_proba"):
                return self.model.predict_proba(texts, rag_features)
            raise NotImplementedError("distilbert_clf 未实现 predict_proba 方法")

        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        raise NotImplementedError(f"{self.model_name} 未实现 predict_proba 方法")

    def get_hyperparams_str(self, training_hparams: dict = None, extra_params: dict = None) -> str:
        """
        由底层模型生成超参数说明字符串。
        """
        if hasattr(self.model, "get_hyperparams_str"):
            return self.model.get_hyperparams_str(training_hparams or {}, extra_params or {})
        return f"Layer2: {self.model_name} (no hyperparams_str)"

