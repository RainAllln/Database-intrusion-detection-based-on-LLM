import os
from datetime import datetime
import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from src import SQLPreprocessor, SQLEmbedder, Layer1Detector, Layer2Classifier

def main():
    # 1. 数据加载与预处理
    data_path = 'data/custom/custom_dataset.csv'
    try:
        df = pd.read_csv(data_path, encoding='utf-16')
    except UnicodeError:
        df = pd.read_csv(data_path, encoding='latin-1')

    preprocessor = SQLPreprocessor()
    df['clean_query'] = df['query'].apply(preprocessor.normalize)
    df['ast_query'] = df['query'].apply(lambda x: " ".join(preprocessor.get_ast_sequence(x)))
    df['Label'] = pd.to_numeric(df['Label'], errors='coerce').fillna(0).astype(int)

    # 2. 特征提取（DistilBERT）
    embedder = SQLEmbedder()
    batch_size = 128 if torch.cuda.is_available() else 32
    embeddings = embedder.get_embeddings(df['ast_query'].values, batch_size=batch_size)

    # 3. 划分训练集和测试集（全流程统一划分）
    train_idx, test_idx = train_test_split(np.arange(len(df)), test_size=0.2, random_state=42)
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    train_embeddings = embeddings[train_idx]
    test_embeddings = embeddings[test_idx]

    # 4. 第一层训练（孤立森林）
    y_train_layer1 = train_df['Label'].apply(lambda x: 1 if x == 1 else 0).values
    X_train_normal = train_embeddings[y_train_layer1 == 0]
    l1_detector = Layer1Detector(model_name="isolation_forest")
    l1_detector.train(X_train_normal)

    # 5. 第二层训练（MLP角色分类）
    num_roles = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 只用训练集中的正常和越权样本训练角色分类
    train_roles_mask = train_df['Label'] != 1
    X_train2 = train_embeddings[train_roles_mask]
    y_train2 = train_df.loc[train_roles_mask, 'role'].values
    X_train2 = torch.tensor(X_train2, dtype=torch.float32)
    y_train2 = torch.tensor(y_train2, dtype=torch.long)
    model = Layer2Classifier(model_name="mlp", num_roles=num_roles).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train2, y_train2), batch_size=32, shuffle=True
    )
    model.train()
    for epoch in range(15):
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    # 6. 测试集整体框架测试
    # 第一层检测
    y_test_layer1 = test_df['Label'].apply(lambda x: 1 if x == 1 else 0).values
    X_test = test_embeddings
    preds, scores = l1_detector.detect(X_test)
    # -1为异常（SQL注入），1为正常
    l1_result = np.array([1 if p == -1 else 0 for p in preds])  # 1:注入, 0:正常

    # 初始化框架输出标签，全部为0（正常）
    framework_pred = np.zeros(len(test_df), dtype=int)
    # 第一层未通过的直接标记为1
    framework_pred[l1_result == 1] = 1

    # 第一层通过的进入第二层
    l2_mask = (l1_result == 0)
    l2_embeddings = X_test[l2_mask]
    l2_roles_true = test_df.loc[l2_mask, 'role'].values
    l2_labels_true = test_df.loc[l2_mask, 'Label'].values

    # 第二层推理
    model.eval()
    with torch.no_grad():
        l2_embeddings_tensor = torch.tensor(l2_embeddings, dtype=torch.float32).to(device)
        outputs = model(l2_embeddings_tensor)
        _, l2_roles_pred = torch.max(outputs, 1)
        l2_roles_pred = l2_roles_pred.cpu().numpy()

    # 角色不匹配的标记为2（伪装攻击）
    l2_result = (l2_roles_pred == l2_roles_true)
    # 框架输出标签：角色不匹配的为2，匹配的为0
    framework_pred[l2_mask] = np.where(l2_result, 0, 2)

    # 7. 统计整体准确率（只统计框架判定为0且真实Label为0的样本）
    true_label = test_df['Label'].values
    correct_mask = (framework_pred == 0) & (true_label == 0)
    acc = np.mean(correct_mask)
    print(f"\n=== SQL语句检测整体框架准确率: {acc:.4f} ===")
    print(f"测试集总数: {len(test_df)}, 框架判定为正常且真实正常数: {correct_mask.sum()}")

    print("\n详细分类报告（框架输出标签与真实标签）:")
    print(classification_report(true_label, framework_pred, target_names=['正常', '注入', '伪装']))

if __name__ == "__main__":
    main()
