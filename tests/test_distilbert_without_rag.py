import os
import sys
import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import torch.nn as nn

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from src.preprocess import SQLPreprocessor
from src.feature import SQLEmbedder


class DistilBertLinearClassifier(nn.Module):
    """DistilBERT 句向量 + 单层线性分类头"""
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


def test_distilbert_without_rag():
    # 1. 加载数据（与 MLP 测试保持一致）
    data_path = os.path.join(project_root, 'data', 'custom', 'complex_dataset_v2.csv')
    print(f"正在读取数据集: {data_path}")
    try:
        df = pd.read_csv(data_path, encoding='utf-8')
    except (UnicodeError, FileNotFoundError, Exception):
        try:
            df = pd.read_csv(data_path, encoding='gbk')
        except Exception:
            df = pd.read_csv(data_path, encoding='latin-1')

    df = df.dropna(subset=['query', 'role', 'Label']).reset_index(drop=True)
    print(f"有效样本数: {len(df)}")

    num_roles = 8

    # 2. 特征提取：DistilBERT + clean_query，不微调，不用 RAG
    preprocessor = SQLPreprocessor()
    embedder = SQLEmbedder(extractor_type="distilbert")

    print("正在提取 DistilBERT 语义特征 (基于 clean_query)，用于『DistilBERT+线性头』8 角色分类测试...")
    df['clean_query'] = df['query'].astype(str).apply(preprocessor.normalize)

    X_embeddings = embedder.get_embeddings(df['clean_query'].values, batch_size=128)
    if X_embeddings is None:
        raise RuntimeError("get_embeddings 返回 None，请检查 DistilBERTFeatureExtractor.get_embeddings 实现。")

    y_roles = df['role'].values.astype(int)

    # 3. 划分训练集 / 测试集（按角色分层）
    X_train, X_test, y_train, y_test = train_test_split(
        X_embeddings,
        y_roles,
        test_size=0.2,
        random_state=42,
        stratify=y_roles,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

    # 4. 初始化 DistilBERT 线性分类模型（输入维度=句向量维度）
    input_dim = X_train.shape[1]
    model = DistilBertLinearClassifier(input_dim=input_dim, num_classes=num_roles).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    train_loader = DataLoader(
        TensorDataset(X_train_tensor, y_train_tensor),
        batch_size=32,
        shuffle=True,
    )

    print(f"开始训练 DistilBERT+线性头 8 角色分类模型（无 RAG 特征，无 DistilBERT 微调），Epochs: 15, Device: {device}")
    for epoch in range(15):
        model.train()
        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch + 1}/15], Loss: {avg_loss:.4f}")

    # 5. 测试：只看 8 角色预测准确率
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, y_pred = torch.max(outputs, dim=1)

    y_pred_np = y_pred.cpu().numpy()
    y_true_np = y_test_tensor.cpu().numpy()

    acc = accuracy_score(y_true_np, y_pred_np)
    print("\n=== DistilBERT+线性头（无 RAG 特征、无 DistilBERT 微调）8 角色分类评估报告 ===")
    print(f"角色预测准确率: {acc:.4f}")
    print(classification_report(y_true_np, y_pred_np, target_names=[f'R{i}' for i in range(num_roles)]))

    return acc, (y_true_np, y_pred_np)


if __name__ == "__main__":
    test_distilbert_without_rag()

