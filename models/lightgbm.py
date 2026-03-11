import os
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt

class LightGBMClassifierModel:
    """
    第二层 LightGBM 角色分类模型：
    - 适用于多分类 (num_roles 类)
    - train_model: 训练逻辑
    - predict_proba: 推理逻辑
    - get_hyperparams_str: 记录参数
    - 可输出特征重要性，帮助分析哪些维度 / 哪些 RAG 距离对判定越权最关键
    """
    def __init__(self, input_dim: int, num_roles: int, device=None):
        # device 在树模型中暂不使用，仅保持接口一致
        self.input_dim = input_dim
        self.num_roles = num_roles
        self.device = device
        self.model: lgb.Booster | None = None

    def train_model(self, X_train: np.ndarray, y_train: np.ndarray,
                    training_hparams: dict, output_dir: str = None):
        """
        训练 LightGBM 多分类模型：
        X_train: np.ndarray [N, input_dim]
        y_train: np.ndarray [N]，角色标签 (0..num_roles-1)
        返回 loss_history(list[float])，这里记录若干轮的 multi_logloss
        """
        # 一些默认参数，可通过 training_hparams 覆盖
        params = {
            "objective": "multiclass",
            "num_class": self.num_roles,
            "metric": "multi_logloss",
            "learning_rate": training_hparams.get("learning_rate", 0.05),
            "num_leaves": training_hparams.get("num_leaves", 31),
            "max_depth": training_hparams.get("max_depth", -1),
            "feature_fraction": training_hparams.get("feature_fraction", 0.9),
            "bagging_fraction": training_hparams.get("bagging_fraction", 0.8),
            "bagging_freq": training_hparams.get("bagging_freq", 1),
            "min_data_in_leaf": training_hparams.get("min_data_in_leaf", 20),
            "verbose": -1,
        }
        num_boost_round = training_hparams.get("n_estimators", 200)

        train_data = lgb.Dataset(X_train, label=y_train)

        # 旧版本 lightgbm 不支持 evals_result 关键字参数，这里只使用最基础的接口
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=num_boost_round,
        )

        # 手动构造一个简单的 loss_history：用训练前后两次 loss 近似
        loss_history = []
        try:
            # 训练后使用模型预测，再根据 multi_logloss 定义自己算一次损失
            logits = self.model.predict(X_train)  # shape [N, num_roles]
            logits = np.clip(logits, 1e-12, 1.0)
            # one-hot
            y_onehot = np.eye(self.num_roles)[y_train.astype(int)]
            logloss = -np.mean(np.sum(y_onehot * np.log(logits), axis=1))
            loss_history.append(float(logloss))
        except Exception as e:
            print(f"[LightGBM] 计算训练集 logloss 时出错: {e}")
            loss_history = []

        # 如指定 output_dir，则输出特征重要性图
        if output_dir is not None:
            try:
                os.makedirs(output_dir, exist_ok=True)
                fig, ax = plt.subplots(figsize=(10, 6))
                lgb.plot_importance(self.model, ax=ax, max_num_features=30)
                ax.set_title("Layer2 LightGBM Feature Importance (Top 30)")
                fig.tight_layout()
                fig_path = os.path.join(output_dir, "layer2_lightgbm_feature_importance.png")
                fig.savefig(fig_path)
                plt.close(fig)
            except Exception as e:
                print(f"[LightGBM] 绘制 / 保存特征重要性时出错: {e}")

        return loss_history

    def predict_proba(self, X: np.ndarray):
        """
        推理接口：
        X: np.ndarray [N, input_dim]
        返回:
          y_pred: np.ndarray[int], shape (N,)
          probs:  np.ndarray[float], shape (N, num_roles)
        """
        if self.model is None:
            raise RuntimeError("LightGBM model 尚未训练，请先调用 train_model。")

        probs = self.model.predict(X)  # shape [N, num_roles]
        probs = np.asarray(probs, dtype=np.float32)
        y_pred = np.argmax(probs, axis=1)
        return y_pred, probs

    def get_hyperparams_str(self, training_hparams: dict, extra_params: dict | None = None) -> str:
        """
        返回当前 LightGBM 模型与训练的一些关键信息，用于实验报告。
        """
        extra_params = extra_params or {}
        rag_threshold = extra_params.get("rag_threshold", None)
        num_roles = extra_params.get("num_roles", self.num_roles)

        lines = [
            "Layer2: LightGBMClassifierModel",
            f"  - input_dim: {self.input_dim}",
            f"  - num_roles: {self.num_roles}",
            "  - objective: multiclass",
            f"  - n_estimators: {training_hparams.get('n_estimators', 200)}",
            f"  - learning_rate: {training_hparams.get('learning_rate', 0.05)}",
            f"  - num_leaves: {training_hparams.get('num_leaves', 31)}",
            f"  - max_depth: {training_hparams.get('max_depth', -1)}",
            f"  - feature_fraction: {training_hparams.get('feature_fraction', 0.9)}",
            f"  - bagging_fraction: {training_hparams.get('bagging_fraction', 0.8)}",
            f"  - bagging_freq: {training_hparams.get('bagging_freq', 1)}",
            f"  - min_data_in_leaf: {training_hparams.get('min_data_in_leaf', 20)}",
            f"  - num_roles(param): {num_roles}",
        ]
        if rag_threshold is not None:
            lines.append(f"  - rag_threshold: {rag_threshold}")
        return "\n".join(lines)
