import os
from datetime import datetime
import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from src import SQLPreprocessor, SQLEmbedder, Layer1Detector, Layer2Classifier
# --- 新增: FAISS与RAG辅助函数 ---
import faiss

def build_role_kb_faiss_l2(X_train, y_train, labels_train, num_roles):
    # 仅存储正常SQL（Label==0），每个角色一个FAISS L2索引
    mask = (labels_train == 0)
    kb = {}
    X_np = X_train if isinstance(X_train, np.ndarray) else X_train.cpu().numpy()
    y_np = y_train if isinstance(y_train, np.ndarray) else (y_train.cpu().numpy() if isinstance(y_train, torch.Tensor) else np.array(y_train))
    dim = X_np.shape[1]
    for r in range(num_roles):
        idx = (y_np[mask] == r)
        vecs = X_np[mask][idx]
        index = faiss.IndexFlatL2(dim)
        if vecs.shape[0] > 0:
            index.add(vecs.astype('float32'))
        else:
            index.add(np.zeros((1, dim), dtype='float32'))
        kb[r] = index
    return kb

def get_top1_l2_distances(sql_emb, kb, num_roles):
    emb = sql_emb.astype('float32').reshape(1, -1)
    dists = []
    for r in range(num_roles):
        D, _ = kb[r].search(emb, 1)
        dists.append(float(D[0][0]))
    return np.array(dists)

def main():
    # 1. 数据加载与预处理
    data_path = 'data/custom/custom_dataset.csv'
    try:
        df = pd.read_csv(data_path, encoding='utf-8')
    except Exception:
        try:
            df = pd.read_csv(data_path, encoding='gbk')
        except Exception:
            df = pd.read_csv(data_path, encoding='latin-1')

    preprocessor = SQLPreprocessor()
    df['clean_query'] = df['query'].apply(preprocessor.normalize)
    df['ast_query'] = df['query'].apply(lambda x: " ".join(preprocessor.get_ast_sequence(x)))
    df['Label'] = pd.to_numeric(df['Label'], errors='coerce').fillna(0).astype(int)

    # 2. 特征提取（DistilBERT, 使用AST）
    embedder = SQLEmbedder()
    batch_size = 128 if torch.cuda.is_available() else 32
    embeddings = embedder.get_embeddings(df['ast_query'].values, batch_size=batch_size)

    # 3. 划分训练集和测试集（全流程统一划分）
    train_idx, test_idx = train_test_split(np.arange(len(df)), test_size=0.2, random_state=42)
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    train_embeddings = embeddings[train_idx]
    test_embeddings = embeddings[test_idx]

    # 4. 第一层训练（孤立森林，仅正常样本）
    y_train_layer1 = train_df['Label'].apply(lambda x: 1 if x == 1 else 0).values
    X_train_normal = train_embeddings[y_train_layer1 == 0]
    l1_detector = Layer1Detector(model_name="isolation_forest")
    l1_detector.train(X_train_normal)

    # 5. 第二层训练（MLP+RAG角色分类）
    num_roles = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 训练集中过滤掉注入(Label!=1)，用于角色分类与KB构建
    train_roles_mask = train_df['Label'] != 1
    X_train2 = train_embeddings[train_roles_mask]
    y_train2 = train_df.loc[train_roles_mask, 'role'].values
    labels_train2 = train_df.loc[train_roles_mask, 'Label'].values

    # 构建FAISS角色知识库（仅Label==0）
    role_kb = build_role_kb_faiss_l2(X_train2, y_train2, labels_train2, num_roles)

    # 训练时计算RAG Top-1 L2距离并拼接到embedding
    rag_features_train = []
    for emb in X_train2:
        rag_features_train.append(get_top1_l2_distances(emb, role_kb, num_roles))
    rag_features_train = np.array(rag_features_train)  # (N, num_roles)
    X_train2_rag = np.concatenate([X_train2, rag_features_train], axis=1)

    X_train2_tensor = torch.tensor(X_train2_rag, dtype=torch.float32).to(device)
    y_train2_tensor = torch.tensor(y_train2, dtype=torch.long).to(device)

    model = Layer2Classifier(model_name="mlp", num_roles=num_roles).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train2_tensor, y_train2_tensor), batch_size=32, shuffle=True
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
    l1_result = np.array([1 if p == -1 else 0 for p in preds])  # 1:注入, 0:正常

    # 框架输出标签初始化
    framework_pred = np.zeros(len(test_df), dtype=int)
    framework_pred[l1_result == 1] = 1  # 第一层未通过标记为注入

    # 第一层通过的进入第二层
    l2_mask = (l1_result == 0)
    l2_embeddings = X_test[l2_mask]
    l2_roles_true = test_df.loc[l2_mask, 'role'].values
    l2_labels_true = test_df.loc[l2_mask, 'Label'].values

    # 计算RAG特征，拼接后推理
    rag_features_test = []
    for emb in l2_embeddings:
        rag_features_test.append(get_top1_l2_distances(emb, role_kb, num_roles))
    rag_features_test = np.array(rag_features_test)
    l2_input = np.concatenate([l2_embeddings, rag_features_test], axis=1)

    model.eval()
    with torch.no_grad():
        l2_tensor = torch.tensor(l2_input, dtype=torch.float32).to(device)
        outputs = model(l2_tensor)
        _, l2_roles_pred = torch.max(outputs, 1)
        l2_roles_pred = l2_roles_pred.cpu().numpy()

    # 基于RAG距离与角色匹配的最终判定
    rag_threshold = 0.7
    final_l2_pred = []
    for i, pred_role in enumerate(l2_roles_pred):
        dist = rag_features_test[i][pred_role]
        if (pred_role == l2_roles_true[i]) and (dist < rag_threshold):
            final_l2_pred.append(0)
        else:
            final_l2_pred.append(2)
    framework_pred[l2_mask] = np.array(final_l2_pred)

    # 7. 统计整体准确率与分类报告
    true_label = test_df['Label'].values
    correct_mask = (framework_pred == 0) & (true_label == 0)
    acc = np.mean(correct_mask)
    print(f"\n=== SQL语句检测整体框架准确率: {acc:.4f} ===")
    print(f"测试集总数: {len(test_df)}, 框架判定为正常且真实正常数: {correct_mask.sum()}")

    print("\n详细分类报告（框架输出标签与真实标签）:")
    print(classification_report(true_label, framework_pred, target_names=['正常', '注入', '伪装']))

if __name__ == "__main__":
    main()
