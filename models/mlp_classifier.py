import torch.nn as nn
import torch.nn.functional as F

class MLPClassifier(nn.Module):
    def __init__(self, input_dim=772, num_roles=4):
        super(MLPClassifier, self).__init__()
        # 记录结构超参数，便于 main.py 查询
        self.input_dim = input_dim
        self.num_roles = num_roles
        self.hidden_sizes = [512, 128]
        self.dropout = 0.3

        self.main = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(128, num_roles)
        )

    def forward(self, sql_embedding):
        logits = self.main(sql_embedding)
        return F.softmax(logits, dim=1)

    def get_hyperparams(self, training_hparams: dict | None = None):
        """
        返回MLP结构及训练相关超参数:
        training_hparams: 由外部训练过程提供的dict，例如 {"lr":..., "epochs":..., "batch_size":...}
        """
        h = {
            "input_dim": self.input_dim,
            "hidden_sizes": self.hidden_sizes,
            "dropout": self.dropout,
            "num_roles": self.num_roles,
        }
        if training_hparams is not None:
            h.update(training_hparams)
        return h

    def get_hyperparams_str(
        self,
        training_hparams: dict | None = None,
        rag_params: dict | None = None,
    ) -> str:
        """
        返回“Layer2: MLP + RAG”的完整超参数字符串。
        - training_hparams: lr/epochs/batch_size 等，由外部训练过程提供
        - rag_params: 与第二层配合使用的 RAG 相关超参数（如 num_roles, rag_threshold 等）
        这样 main.py 只负责把两个 dict 传进来，真正如何展示由模型自己决定。
        """
        h = self.get_hyperparams(training_hparams=training_hparams)
        lines = [
            "Layer2: MLP + RAG",
            f"  - input_dim: {h.get('input_dim')}",
            f"  - hidden_sizes: {h.get('hidden_sizes')}",
            f"  - dropout: {h.get('dropout')}",
            f"  - num_roles: {h.get('num_roles')}",
        ]
        # 训练相关
        if "lr" in h or "epochs" in h or "batch_size" in h:
            lines.append(
                f"  - optimizer: Adam, lr: {h.get('lr')}, epochs: {h.get('epochs')}, batch_size: {h.get('batch_size')}"
            )
        # RAG 相关（如果提供）
        if rag_params is not None:
            lines.append("  - RAG params:")
            lines.append(f"      * num_roles: {rag_params.get('num_roles')}")
            lines.append(f"      * rag_threshold: {rag_params.get('rag_threshold')}")
            lines.append(f"      * faiss_metric: {rag_params.get('faiss_metric')}")
        return "\n".join(lines)
