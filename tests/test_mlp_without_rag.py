import os
import sys
import torch
import pandas as pd
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from src import SQLPreprocessor, SQLEmbedder, Layer2Classifier


def test_mlp_without_rag():
    # 1\. 加载数据（与 `main_layer2.py` 保持一致）
    data_path = os.path.join(project_root, 'data', 'custom', 'custom_dataset.csv')
    print(f"正在读取数据集: {data_path}")
    try:
        df = pd.read_csv(data_path, encoding='utf-8')
    except (UnicodeError, FileNotFoundError, Exception):
        try:
            df = pd.read_csv(data_path, encoding='gbk')
        except Exception:
            df = pd.read_csv(data_path, encoding='latin-1')

    # 只保留 Label 0(正常) 和 2(越权)，和第二层任务一致
    df_l2 = df[df['Label'] != 1].reset_index(drop=True)
    print(f"第二层有效样本数: {len(df_l2)}")

    num_roles = 4  # 根据当前数据集中的角色定义

    # 2\. 特征提取：只用 DistilBERT AST embedding，不加任何 RAG 向量
    preprocessor = SQLPreprocessor()
    embedder = SQLEmbedder()

    print("正在提取语义特征 (DistilBERT+AST)，用于纯 MLP 角色分类测试...")
    df_l2['ast_query'] = df_l2['query'].astype(str).apply(
        lambda x: " ".join(preprocessor.get_ast_sequence(x))
    )

    X_embeddings = embedder.get_embeddings(df_l2['ast_query'].values, batch_size=128)
    y_roles = df_l2['role'].values
    labels_all = df_l2['Label'].values  # 方便后面如果想一起分析

    # 3\. 划分训练集 / 测试集
    X_train, X_test, y_train, y_test, labels_train, labels_test = train_test_split(
        X_embeddings,
        y_roles,
        labels_all,
        test_size=0.2,
        random_state=42,
        stratify=y_roles  # 尽量保证各角色比例一致
    )

    # 转为 Tensor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

    # 4\. 只用纯 embedding 训练 MLP（不拼接 RAG 特征）
    model = Layer2Classifier(model_name="mlp", num_roles=num_roles).to(device)

    # 新增：获取模型期望的输入维度并在运行时对输入做零填充/截断
    def get_model_input_dim(m):
        for module in m.modules():
            if isinstance(module, torch.nn.Linear):
                return module.in_features
        return None

    def ensure_input_dim(x, m):
        """
        将输入 x 调整为模型期望的列数：
        - 若 x.shape[1] < needed：在列方向右侧用 0 填充
        - 若 x.shape[1] > needed：截断到 needed
        保持 device 和 dtype 不变。
        """
        needed = get_model_input_dim(m)
        if needed is None:
            return x
        b, feat = x.shape[0], x.shape[1]
        if feat == needed:
            return x
        if feat < needed:
            diff = needed - feat
            pad = torch.zeros((b, diff), dtype=x.dtype, device=x.device)
            return torch.cat([x, pad], dim=1)
        # feat > needed
        return x[:, :needed].contiguous()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    train_loader = DataLoader(
        TensorDataset(X_train_tensor, y_train_tensor),
        batch_size=32,
        shuffle=True
    )

    print(f"开始训练纯 MLP 角色分类模型（无 RAG 特征），Epochs: 15, Device: {device}")
    model.train()
    for epoch in range(15):
        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            # 在送入模型前调整维度
            batch_x = ensure_input_dim(batch_x, model)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch + 1}/15], Loss: {avg_loss:.4f}")

    # 5\. 测试：只看角色预测准确率
    model.eval()
    with torch.no_grad():
        X_test_adj = ensure_input_dim(X_test_tensor, model)
        outputs = model(X_test_adj)
        _, y_pred = torch.max(outputs, 1)
        y_pred = y_pred.cpu().numpy()
        y_true = y_test_tensor.cpu().numpy()

    acc = accuracy_score(y_true, y_pred)
    print("\n=== 纯 MLP（无 RAG 特征）角色分类评估报告 ===")
    print(f"角色预测准确率: {acc:.4f}")
    print(classification_report(y_true, y_pred, target_names=[f'R{i}' for i in range(num_roles)]))

    # 返回结果，方便在别的脚本里对比
    return acc, (y_true, y_pred)


if __name__ == "__main__":
    test_mlp_without_rag()
