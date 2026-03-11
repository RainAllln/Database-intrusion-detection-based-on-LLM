import os
import sys
import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import torch.nn as nn
import numpy as np  # NEW

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from src.preprocess import SQLPreprocessor
from src.feature import SQLEmbedder

# NEW: 使用你现成的 FAISS RAG 逻辑
from src.rag import build_role_knowledge_base_faiss_l2, get_enhanced_rag_features


class DistilBertLinearClassifier(nn.Module):
    """（拼接后）特征向量 + 单层线性分类头"""
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


def _extract_rag_features_from_kb(X_emb_np: np.ndarray, kb, num_roles: int, k: int = 5) -> np.ndarray:
    """给定 DistilBERT embedding (N,D)，用 FAISS KB 抽取 RAG 特征 (N, rag_dim)"""
    feats = []
    for i in range(X_emb_np.shape[0]):
        feats.append(get_enhanced_rag_features(X_emb_np[i], kb, num_roles=num_roles, k=k))
    return np.asarray(feats, dtype=np.float32)


def test_distilbert_with_rag(
    rag_weight: float = 3.0,   # 默认提高一点
    rag_k: int = 5,
    batch_size_embed: int = 128,
    epochs: int = 15,
):
    # 1. 加载数据
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

    # 2. 预处理
    preprocessor = SQLPreprocessor()
    df['clean_query'] = df['query'].astype(str).apply(preprocessor.normalize)

    # 3. 先提 DistilBERT embedding（RAG 特征将基于该 embedding + FAISS KB 计算）
    embedder_distilbert = SQLEmbedder(extractor_type="distilbert")

    print("正在提取 DistilBERT 语义特征...")
    X_d = embedder_distilbert.get_embeddings(df['clean_query'].values, batch_size=batch_size_embed)
    if X_d is None:
        raise RuntimeError("DistilBERT get_embeddings 返回 None，请检查实现。")

    y_roles = df['role'].values.astype(int)
    labels_global = df['Label'].values.astype(int)  # NEW: 用于 KB 仅保留 Label==0

    # 4. 先划分，再用训练集构建 FAISS KB（避免数据泄漏）
    Xd_train, Xd_test, y_train, y_test, lab_train, lab_test = train_test_split(
        X_d,
        y_roles,
        labels_global,
        test_size=0.2,
        random_state=42,
        stratify=y_roles,
    )

    # 5. 构建每个 role 的 FAISS 知识库（仅 Label==0 的正常样本）
    print("正在基于训练集(Label==0)构建 FAISS 角色知识库...")
    kb = build_role_knowledge_base_faiss_l2(
        X_train=torch.tensor(Xd_train, dtype=torch.float32),
        y_train=torch.tensor(y_train, dtype=torch.long),
        labels_train=torch.tensor(lab_train, dtype=torch.long),
        num_roles=num_roles,
    )

    # 6. 抽取 RAG 增强特征（对 train/test 都做）
    print(f"正在抽取 RAG 特征(get_enhanced_rag_features, k={rag_k})...")
    Xr_train = _extract_rag_features_from_kb(np.asarray(Xd_train, dtype=np.float32), kb, num_roles=num_roles, k=rag_k)
    Xr_test = _extract_rag_features_from_kb(np.asarray(Xd_test, dtype=np.float32), kb, num_roles=num_roles, k=rag_k)

    # 7. 加权拼接（提高 RAG 权重）
    #    让线性头看到 [distilbert, rag_weight * rag]
    X_train = np.concatenate([Xd_train.astype(np.float32), (rag_weight * Xr_train).astype(np.float32)], axis=1)
    X_test = np.concatenate([Xd_test.astype(np.float32), (rag_weight * Xr_test).astype(np.float32)], axis=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

    # 8. 线性分类头（输入维度=拼接后的维度）
    input_dim = X_train.shape[1]
    model = DistilBertLinearClassifier(input_dim=input_dim, num_classes=num_roles).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    train_loader = DataLoader(
        TensorDataset(X_train_tensor, y_train_tensor),
        batch_size=32,
        shuffle=True,
    )

    print(
        f"开始训练 DistilBERT+FAISS-RAG(拼接, rag_weight={rag_weight}, k={rag_k}) + 线性头 8 角色分类模型，"
        f"Epochs: {epochs}, Device: {device}"
    )
    for epoch in range(epochs):
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
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")

    # 9. 测试
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, y_pred = torch.max(outputs, dim=1)

    y_pred_np = y_pred.cpu().numpy()
    y_true_np = y_test_tensor.cpu().numpy()

    acc = accuracy_score(y_true_np, y_pred_np)
    print("\n=== DistilBERT+FAISS-RAG(拼接) + 线性头 8 角色分类评估报告 ===")
    print(f"rag_weight: {rag_weight}, rag_k: {rag_k}")
    print(f"角色预测准确率: {acc:.4f}")
    print(classification_report(y_true_np, y_pred_np, target_names=[f'R{i}' for i in range(num_roles)]))

    return acc, (y_true_np, y_pred_np)


if __name__ == "__main__":
    test_distilbert_with_rag()
