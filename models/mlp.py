import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class MLPClassifierModel(nn.Module):
    """
    第二层 MLP 角色分类模型：
    - 内部包含网络结构
    - train_model: 训练逻辑
    - predict_proba: 推理逻辑
    - get_hyperparams_str: 记录参数
    """
    def __init__(self, input_dim: int, num_roles: int, device: torch.device):
        super().__init__()
        self.input_dim = input_dim
        self.num_roles = num_roles
        self.device = device

        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_roles),
        ).to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def train_model(self, X_train: np.ndarray, y_train: np.ndarray,
                    training_hparams: dict, output_dir: str = None):
        """
        训练 MLP：
        X_train: np.ndarray [N, input_dim]
        y_train: np.ndarray [N]
        返回 loss_history(list[float])
        """
        lr = training_hparams.get("lr", 1e-3)
        epochs = training_hparams.get("epochs", 15)
        batch_size = training_hparams.get("batch_size", 32)

        X_tensor = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y_train, dtype=torch.long).to(self.device)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        self.train()
        loss_history = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.forward(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(loader)
            loss_history.append(avg_loss)
            if (epoch + 1) % max(1, epochs // 3) == 0:
                print(f"[MLP] Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        return loss_history

    def predict_proba(self, X: np.ndarray):
        """
        推理接口：
        X: np.ndarray [N, input_dim]
        返回:
          y_pred: np.ndarray[int], shape (N,)
          probs:  np.ndarray[float], shape (N, num_roles)
        """
        self.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            logits = self.forward(X_tensor)
            probs = torch.softmax(logits, dim=1)
            y_pred = torch.argmax(probs, dim=1)

        return y_pred.cpu().numpy(), probs.cpu().numpy()

    def get_hyperparams_str(self, training_hparams: dict, extra_params: dict | None = None) -> str:
        lr = training_hparams.get("lr", 1e-3)
        epochs = training_hparams.get("epochs", 15)
        batch_size = training_hparams.get("batch_size", 32)
        extra_params = extra_params or {}
        rag_threshold = extra_params.get("rag_threshold", None)
        num_roles = extra_params.get("num_roles", self.num_roles)

        lines = [
            "Layer2: MLPClassifierModel",
            f"  - input_dim: {self.input_dim}",
            f"  - num_roles: {self.num_roles}",
            "  - architecture: [input -> 512 -> 256 -> num_roles] with ReLU + Dropout(0.3)",
            f"  - lr: {lr}",
            f"  - epochs: {epochs}",
            f"  - batch_size: {batch_size}",
            f"  - device: {self.device}",
            f"  - num_roles(param): {num_roles}",
        ]
        if rag_threshold is not None:
            lines.append(f"  - rag_threshold: {rag_threshold}")
        return "\n".join(lines)
