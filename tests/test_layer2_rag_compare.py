import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from src.feature import SQLEmbedder
from src.preprocess import SQLPreprocessor
from src import Layer2Classifier

def l2_normalize(vecs):
    norm = np.linalg.norm(vecs, axis=1, keepdims=True)
    norm[norm == 0] = 1
    return vecs / norm

def build_role_knowledge_base_faiss_cosine(X_train, y_train, labels_train, num_roles):
    import faiss
    mask = (labels_train == 0)
    kb = {}
    X_train_np = X_train.cpu().numpy() if isinstance(X_train, torch.Tensor) else X_train
    y_train_np = y_train.cpu().numpy() if isinstance(y_train, torch.Tensor) else y_train
    dim = X_train_np.shape[1]
    for r in range(num_roles):
        idx = (y_train_np[mask] == r)
        vecs = X_train_np[mask][idx]
        if vecs.shape[0] == 0:
            index = faiss.IndexFlatIP(dim)
            index.add(np.zeros((1, dim), dtype='float32'))
        else:
            vecs = l2_normalize(vecs.astype('float32'))
            index = faiss.IndexFlatIP(dim)
            index.add(vecs)
        kb[r] = index
    return kb

def build_role_knowledge_base_faiss_l2(X_train, y_train, labels_train, num_roles):
    import faiss
    mask = (labels_train == 0)
    kb = {}
    X_train_np = X_train.cpu().numpy() if isinstance(X_train, torch.Tensor) else X_train
    y_train_np = y_train.cpu().numpy() if isinstance(y_train, torch.Tensor) else y_train
    dim = X_train_np.shape[1]
    for r in range(num_roles):
        idx = (y_train_np[mask] == r)
        vecs = X_train_np[mask][idx]
        if vecs.shape[0] == 0:
            index = faiss.IndexFlatL2(dim)
            index.add(np.zeros((1, dim), dtype='float32'))
        else:
            index = faiss.IndexFlatL2(dim)
            index.add(vecs.astype('float32'))
        kb[r] = index
    return kb

def get_topk_cosine_similarities_faiss(sql_emb, kb, num_roles, k=3):
    emb = sql_emb.astype('float32').reshape(1, -1)
    emb = l2_normalize(emb)
    sims = []
    for r in range(num_roles):
        D, _ = kb[r].search(emb, k)
        sim = float(np.mean(D[0]))
        sims.append(sim)
    return np.array(sims)

def get_topk_l2_similarities_faiss(sql_emb, kb, num_roles, k=3):
    emb = sql_emb.astype('float32').reshape(1, -1)
    sims = []
    for r in range(num_roles):
        D, _ = kb[r].search(emb, k)
        sim = float(np.mean(D[0]))
        sims.append(sim)
    return np.array(sims)

def compute_dynamic_thresholds(role_kb, X_train, y_train, labels_train, num_roles, k=3, metric='cosine'):
    thresholds = {}
    X_train_np = X_train.cpu().numpy() if isinstance(X_train, torch.Tensor) else X_train
    y_train_np = y_train.cpu().numpy() if isinstance(y_train, torch.Tensor) else y_train
    labels_train_np = labels_train if isinstance(labels_train, np.ndarray) else np.array(labels_train)
    for r in range(num_roles):
        idx = (labels_train_np == 0) & (y_train_np == r)
        vecs = X_train_np[idx]
        if len(vecs) == 0:
            thresholds[r] = 0.0
            continue
        sims = []
        for emb in vecs:
            if metric == 'cosine':
                sim = get_topk_cosine_similarities_faiss(emb, role_kb, num_roles, k=k)[r]
            else:
                sim = get_topk_l2_similarities_faiss(emb, role_kb, num_roles, k=k)[r]
            sims.append(sim)
        mu = np.mean(sims)
        sigma = np.std(sims)
        p5 = np.percentile(sims, 5)
        thresholds[r] = min(mu - 3 * sigma, p5)
    return thresholds

def evaluate_layer2(X_test, rag_features, y_test, test_roles, test_labels, model, rag_thresholds=None, metric='cosine', static_threshold=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if rag_features is not None:
        X_test_rag = torch.tensor(np.concatenate([X_test.cpu().numpy(), rag_features], axis=1), dtype=torch.float32).to(device)
    else:
        X_test_rag = X_test.to(device)
    with torch.no_grad():
        test_outputs = model(X_test_rag)
        _, y_pred = torch.max(test_outputs, 1)
        y_pred = y_pred.cpu().numpy()

    final_pred = []
    for i, (true_role, pred_role) in enumerate(zip(test_roles, y_pred)):
        # 先判定角色是否匹配
        if pred_role != true_role:
            final_pred.append(2)
            continue
        # 角色匹配后，用预测角色的 RAG 相似度进行阈值判定
        if rag_features is not None:
            if rag_thresholds is not None:
                rag_sim = rag_features[i][pred_role]
                threshold = rag_thresholds[pred_role]
                if metric == 'cosine':
                    if rag_sim < threshold:
                        final_pred.append(2)
                        continue
                else:
                    # L2 距离，大于阈值视为不相似
                    if rag_sim > threshold:
                        final_pred.append(2)
                        continue
            elif static_threshold is not None:
                rag_sim = rag_features[i][pred_role]
                if metric == 'cosine':
                    if rag_sim < static_threshold:
                        final_pred.append(2)
                        continue
                else:
                    if rag_sim > static_threshold:
                        final_pred.append(2)
                        continue
        # 通过
        final_pred.append(0)

    final_pred = np.array(final_pred)
    mask = (final_pred == 0)
    acc = np.mean(test_labels[mask] == 0) if np.sum(mask) > 0 else 0
    not_pass_mask = (final_pred != 0)
    acc_label2 = np.mean(test_labels[not_pass_mask] == 2) if np.sum(not_pass_mask) > 0 else 0
    return acc, acc_label2, np.sum(mask), np.sum(not_pass_mask)

def plot_confusion(cm, title, save_path):
    plt.figure(figsize=(4, 4))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['0', '2'])
    plt.yticks(tick_marks, ['0', '2'])
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"混淆矩阵已保存: {save_path}")

def main():
    data_path = os.path.join(project_root, 'data', 'custom', 'custom_dataset.csv')
    print(f"读取数据: {data_path}")
    df = pd.read_csv(data_path, encoding='utf-8')
    num_roles = 4
    df_l2 = df[df['Label'] != 1].reset_index(drop=True)
    preprocessor = SQLPreprocessor()
    embedder = SQLEmbedder()
    df_l2['ast_query'] = df_l2['query'].astype(str).apply(lambda x: " ".join(preprocessor.get_ast_sequence(x)))
    X_embeddings = embedder.get_embeddings(df_l2['ast_query'].values, batch_size=64)
    y_roles = torch.tensor(df_l2['role'].values).long()
    # 将原始 query 一并划分，确保与测试集顺序对齐
    queries = df_l2['query'].astype(str).values
    X_train, X_test, y_train, y_test, labels_train, labels_test, queries_train, queries_test = train_test_split(
        X_embeddings, y_roles, df_l2['Label'].values, queries, test_size=0.2, random_state=42
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not isinstance(X_train, torch.Tensor):
        X_train = torch.tensor(X_train, dtype=torch.float32)
    if not isinstance(X_test, torch.Tensor):
        X_test = torch.tensor(X_test, dtype=torch.float32)
    if not isinstance(y_train, torch.Tensor):
        y_train = torch.tensor(y_train, dtype=torch.long)
    if not isinstance(y_test, torch.Tensor):
        y_test = torch.tensor(y_test, dtype=torch.long)

    test_roles = y_test.cpu().numpy()

    os.makedirs('rag_compare_results', exist_ok=True)

    # 1. 不引入RAG特征
    print("\n【不引入RAG特征】")
    model_plain = Layer2Classifier(model_name="mlp", num_roles=num_roles, input_dim=768).to(device)
    optimizer = torch.optim.Adam(model_plain.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    model_plain.train()
    for epoch in range(10):
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model_plain(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    model_plain.eval()
    acc_plain, acc_label2_plain, n_pass_plain, n_not_pass_plain = evaluate_layer2(
        X_test, None, y_test, test_roles, labels_test, model_plain
    )
    print(f"最终通过语句数: {n_pass_plain} / {len(labels_test)}")
    print(f"最终准确率（通过语句与Label=0的重合度）: {acc_plain:.4f}")
    print(f"未通过语句数: {n_not_pass_plain} / {len(labels_test)}")
    print(f"未通过语句与Label=2的重合度: {acc_label2_plain:.4f}")

    # 混淆矩阵绘制与保存
    with torch.no_grad():
        test_outputs = model_plain(X_test.to(device))
        _, y_pred_plain = torch.max(test_outputs, 1)
        y_pred_plain = y_pred_plain.cpu().numpy()
    final_pred_plain = []
    for i, (true_role, pred_role) in enumerate(zip(test_roles, y_pred_plain)):
        if pred_role != true_role:
            final_pred_plain.append(2)
        else:
            final_pred_plain.append(0)
    final_pred_plain = np.array(final_pred_plain)
    cm_plain = confusion_matrix(labels_test, final_pred_plain, labels=[0,2])
    plot_confusion(cm_plain, "Confusion Matrix (No RAG)", "rag_compare_results/plot_confusion_plain.png")

    # 1.b 不引入RAG特征 + 静态危险关键字规则
    print("\n【不引入RAG特征 + 静态关键字规则：DELETE/DROP/TRUNCATE 直接判为伪装攻击】")
    import re
    dangerous_pattern = re.compile(r'\b(delete|drop|truncate)\b', flags=re.IGNORECASE)

    with torch.no_grad():
        test_outputs = model_plain(X_test.to(device))
        _, y_pred_plain_rules = torch.max(test_outputs, 1)
        y_pred_plain_rules = y_pred_plain_rules.cpu().numpy()

    final_pred_plain_rules = []
    for i, (true_role, pred_role) in enumerate(zip(test_roles, y_pred_plain_rules)):
        # 若命中危险关键字，直接判为伪装攻击
        if dangerous_pattern.search(queries_test[i]):
            final_pred_plain_rules.append(2)
            continue
        # 否则沿用“MLP角色匹配即通过”的逻辑
        if pred_role != true_role:
            final_pred_plain_rules.append(2)
        else:
            final_pred_plain_rules.append(0)
    final_pred_plain_rules = np.array(final_pred_plain_rules)

    cm_plain_rules = confusion_matrix(labels_test, final_pred_plain_rules, labels=[0,2])
    plot_confusion(cm_plain_rules, "Confusion Matrix (No RAG + Static Rules)", "rag_compare_results/plot_confusion_plain_rules.png")

    # 2. 引入RAG特征（静态阈值，L2距离，Top-1）
    print("\n【RAG特征：FAISS L2距离，静态阈值0.7，Top-1】")
    role_kb_l2 = build_role_knowledge_base_faiss_l2(X_train, y_train, labels_train, num_roles)
    rag_features_train_l2 = []
    for emb in X_train.cpu().numpy():
        rag_features_train_l2.append(get_topk_l2_similarities_faiss(emb, role_kb_l2, num_roles, k=1))
    rag_features_train_l2 = np.array(rag_features_train_l2)
    X_train_rag_l2 = torch.tensor(np.concatenate([X_train.cpu().numpy(), rag_features_train_l2], axis=1), dtype=torch.float32).to(device)
    model_rag_l2 = Layer2Classifier(model_name="mlp", num_roles=num_roles, input_dim=772).to(device)
    optimizer = torch.optim.Adam(model_rag_l2.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    train_loader_rag_l2 = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train_rag_l2, y_train), batch_size=32, shuffle=True)
    model_rag_l2.train()
    for epoch in range(10):
        for batch_x, batch_y in train_loader_rag_l2:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model_rag_l2(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    model_rag_l2.eval()
    rag_features_l2 = []
    for emb in X_test.cpu().numpy():
        rag_features_l2.append(get_topk_l2_similarities_faiss(emb, role_kb_l2, num_roles, k=1))
    rag_features_l2 = np.array(rag_features_l2)
    acc_rag_l2, acc_label2_rag_l2, n_pass_rag_l2, n_not_pass_rag_l2 = evaluate_layer2(
        X_test, rag_features_l2, y_test, test_roles, labels_test, model_rag_l2, metric='l2', static_threshold=0.7
    )
    print(f"最终通过语句数: {n_pass_rag_l2} / {len(labels_test)}")
    print(f"最终准确率（通过语句与Label=0的重合度）: {acc_rag_l2:.4f}")
    print(f"未通过语句数: {n_not_pass_rag_l2} / {len(labels_test)}")
    print(f"未通过语句与Label=2的重合度: {acc_label2_rag_l2:.4f}")

    # 混淆矩阵绘制与保存
    with torch.no_grad():
        X_test_rag_l2 = torch.tensor(np.concatenate([X_test.cpu().numpy(), rag_features_l2], axis=1), dtype=torch.float32).to(device)
        test_outputs = model_rag_l2(X_test_rag_l2)
        _, y_pred_rag_l2 = torch.max(test_outputs, 1)
        y_pred_rag_l2 = y_pred_rag_l2.cpu().numpy()
    final_pred_rag_l2 = []
    for i, (true_role, pred_role) in enumerate(zip(test_roles, y_pred_rag_l2)):
        # 先角色匹配
        if pred_role != true_role:
            final_pred_rag_l2.append(2)
            continue
        # 再用预测角色的 Top-1 L2 距离进行阈值判定（距离 < 0.7 才通过）
        rag_dist = rag_features_l2[i][pred_role]
        if rag_dist > 0.7:
            final_pred_rag_l2.append(2)
        else:
            final_pred_rag_l2.append(0)
    final_pred_rag_l2 = np.array(final_pred_rag_l2)
    cm_rag_l2 = confusion_matrix(labels_test, final_pred_rag_l2, labels=[0,2])
    plot_confusion(cm_rag_l2, "Confusion Matrix (RAG L2 Static)", "rag_compare_results/plot_confusion_rag_l2.png")

    # 3. 引入RAG特征（余弦相似度、动态阈值、Top-K均值）
    print("\n【RAG特征：余弦相似度，动态阈值，Top-K均值】")
    role_kb_cos = build_role_knowledge_base_faiss_cosine(X_train, y_train, labels_train, num_roles)
    rag_features_train_cos = []
    for emb in X_train.cpu().numpy():
        rag_features_train_cos.append(get_topk_cosine_similarities_faiss(emb, role_kb_cos, num_roles, k=3))
    rag_features_train_cos = np.array(rag_features_train_cos)
    X_train_rag_cos = torch.tensor(np.concatenate([X_train.cpu().numpy(), rag_features_train_cos], axis=1), dtype=torch.float32).to(device)
    model_rag_cos = Layer2Classifier(model_name="mlp", num_roles=num_roles, input_dim=772).to(device)
    optimizer = torch.optim.Adam(model_rag_cos.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    train_loader_rag_cos = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train_rag_cos, y_train), batch_size=32, shuffle=True)
    model_rag_cos.train()
    for epoch in range(10):
        for batch_x, batch_y in train_loader_rag_cos:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model_rag_cos(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    model_rag_cos.eval()
    rag_features_cos = []
    for emb in X_test.cpu().numpy():
        rag_features_cos.append(get_topk_cosine_similarities_faiss(emb, role_kb_cos, num_roles, k=3))
    rag_features_cos = np.array(rag_features_cos)
    rag_thresholds_cos = compute_dynamic_thresholds(role_kb_cos, X_train, y_train, labels_train, num_roles, k=3, metric='cosine')
    acc_rag_cos, acc_label2_rag_cos, n_pass_rag_cos, n_not_pass_rag_cos = evaluate_layer2(
        X_test, rag_features_cos, y_test, test_roles, labels_test, model_rag_cos, rag_thresholds=rag_thresholds_cos, metric='cosine'
    )
    print(f"最终通过语句数: {n_pass_rag_cos} / {len(labels_test)}")
    print(f"最终准确率（通过语句与Label=0的重合度）: {acc_rag_cos:.4f}")
    print(f"未通过语句数: {n_not_pass_rag_cos} / {len(labels_test)}")
    print(f"未通过语句与Label=2的重合度: {acc_label2_rag_cos:.4f}")

    # 混淆矩阵绘制与保存
    with torch.no_grad():
        X_test_rag_cos = torch.tensor(np.concatenate([X_test.cpu().numpy(), rag_features_cos], axis=1), dtype=torch.float32).to(device)
        test_outputs = model_rag_cos(X_test_rag_cos)
        _, y_pred_rag_cos = torch.max(test_outputs, 1)
        y_pred_rag_cos = y_pred_rag_cos.cpu().numpy()
    final_pred_rag_cos = []
    for i, (true_role, pred_role) in enumerate(zip(test_roles, y_pred_rag_cos)):
        # 先角色匹配
        if pred_role != true_role:
            final_pred_rag_cos.append(2)
            continue
        # 再用预测角色的 Top-K 余弦相似度与动态阈值比较（相似度 >= 阈值 才通过）
        rag_sim = rag_features_cos[i][pred_role]
        threshold = rag_thresholds_cos[pred_role]
        if rag_sim < threshold:
            final_pred_rag_cos.append(2)
        else:
            final_pred_rag_cos.append(0)
    final_pred_rag_cos = np.array(final_pred_rag_cos)
    cm_rag_cos = confusion_matrix(labels_test, final_pred_rag_cos, labels=[0,2])
    plot_confusion(cm_rag_cos, "Confusion Matrix (RAG Cosine Dynamic)", "rag_compare_results/plot_confusion_rag_cos.png")

if __name__ == "__main__":
    main()

