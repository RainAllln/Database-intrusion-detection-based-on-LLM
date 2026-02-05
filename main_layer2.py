import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from src import SQLPreprocessor, SQLEmbedder, Layer2Classifier
from sklearn.metrics.pairwise import cosine_similarity
import faiss
from src.utils import (
    plot_confusion_matrix, plot_loss_curve, write_classification_report, write_detail_log
)

def build_role_knowledge_base_faiss(X_train, y_train, labels_train, num_roles):
    # 只存储正常SQL（Label==0），每个角色一个FAISS索引
    mask = (labels_train == 0)
    kb = {}
    X_train_np = X_train.cpu().numpy() if isinstance(X_train, torch.Tensor) else X_train
    y_train_np = y_train.cpu().numpy() if isinstance(y_train, torch.Tensor) else y_train
    dim = X_train_np.shape[1]
    for r in range(num_roles):
        idx = (y_train_np[mask] == r)
        vecs = X_train_np[mask][idx]
        if vecs.shape[0] == 0:
            # 空索引，填充一个全零向量
            index = faiss.IndexFlatL2(dim)
            index.add(np.zeros((1, dim), dtype='float32'))
        else:
            index = faiss.IndexFlatL2(dim)
            index.add(vecs.astype('float32'))
        kb[r] = index
    return kb

def get_max_similarities_faiss(sql_emb, kb, num_roles):
    # 对每个角色知识库FAISS索引，计算最大相似度（L2距离转相似度）
    sims = []
    emb = sql_emb.astype('float32').reshape(1, -1)
    for r in range(num_roles):
        D, _ = kb[r].search(emb, 1)  # D: (1,1) 最小距离
        # 距离转相似度（越小越相似，这里用负距离或exp(-d)都可）
        sim = float(np.exp(-D[0][0]))
        sims.append(sim)
    return np.array(sims)

def main_layer2():
    # --- 1. 加载与预处理  ---
    data_path = 'data/custom/custom_dataset.csv'

    print(f"正在读取数据集: {data_path}")
    try:
        df = pd.read_csv(data_path, encoding='utf-8')
    except (UnicodeError, FileNotFoundError, Exception):
        try:
            df = pd.read_csv(data_path, encoding='gbk')
        except:
            df = pd.read_csv(data_path, encoding='latin-1')

    # 第二层只处理非注入语句 (Label 0: 正常, Label 2: 越权)
    # Label 1 是注入攻击，由第一层负责
    print("筛选数据: 过滤掉 Label 1 (SQL注入)，只保留正常(0)和越权(2)...")
    df_l2 = df[df['Label'] != 1].reset_index(drop=True)

    print(f"第二层有效样本数: {len(df_l2)}")

    # --- 2. 创建实验文件夹---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 假设 role 总是 0, 1, 2, 3，所以 num_roles=4
    # 可以用 len(df_l2['role'].unique())
    num_roles = 4

    # 命名规范：exp2_[时间戳]
    exp_folder = f"exp2_{timestamp}"
    output_dir = os.path.join("notebooks", exp_folder)
    os.makedirs(output_dir, exist_ok=True)
    print(f"第二层实验结果将保存至: {output_dir}")

    # --- 3. 特征提取 ---
    preprocessor = SQLPreprocessor()
    embedder = SQLEmbedder()

    print("正在提取语义特征 (DistilBERT)。")
    # 简单清洗
    df_l2['clean_query'] = df_l2['query'].astype(str).apply(preprocessor.normalize)
    # 新增：AST展平SQL
    df_l2['ast_query'] = df_l2['query'].astype(str).apply(lambda x: " ".join(preprocessor.get_ast_sequence(x)))

    # 提取 Embedding
    # 可选：用AST特征或normalize特征
    # X_embeddings = embedder.get_embeddings(df_l2['clean_query'].values, batch_size=128)
    X_embeddings = embedder.get_embeddings(df_l2['ast_query'].values, batch_size=128)
    # y 标签在这里是 role，训练模型去识别这个语句属于哪个角色
    y_roles = torch.tensor(df_l2['role'].values).long()

    # --- 4. 划分数据集 ---
    X_train, X_test, y_train, y_test, labels_train, labels_test = train_test_split(
        X_embeddings, y_roles, df_l2['Label'].values, test_size=0.2, random_state=42
    )

    # 转换为 Tensor
    if not isinstance(X_train, torch.Tensor):
        X_train = torch.tensor(X_train, dtype=torch.float32)
    if not isinstance(X_test, torch.Tensor):
        X_test = torch.tensor(X_test, dtype=torch.float32)
    if not isinstance(y_train, torch.Tensor):
        y_train = torch.tensor(y_train, dtype=torch.long)
    if not isinstance(y_test, torch.Tensor):
        y_test = torch.tensor(y_test, dtype=torch.long)

    # --- 设备定义提前 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- RAG知识库构建（FAISS） ---
    print("正在用FAISS构建RAG角色知识库...")
    role_kb = build_role_knowledge_base_faiss(X_train, y_train, labels_train, num_roles)

    # === 修正：训练时也拼接RAG特征 ===
    rag_features_train = []
    for emb in X_train.cpu().numpy():
        rag_features_train.append(get_max_similarities_faiss(emb, role_kb, num_roles))
    rag_features_train = np.array(rag_features_train)  # shape: (N, 4)
    X_train_rag = torch.tensor(np.concatenate([X_train.cpu().numpy(), rag_features_train], axis=1), dtype=torch.float32).to(device)

    # --- 5. 模型训练 ---
    model = Layer2Classifier(model_name="mlp", num_roles=num_roles).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    train_loader = DataLoader(TensorDataset(X_train_rag, y_train), batch_size=32, shuffle=True)

    print(f"正在训练角色行为模型 (Epochs: 15, Device: {device})...")
    model.train()
    loss_history = []

    for epoch in range(15):
        epoch_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_loss)
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch + 1}/15], Loss: {avg_loss:.4f}")

    # 保存 Loss 曲线
    plot_loss_curve(loss_history, output_dir, filename='training_loss.png', title='Layer 2 Training Loss')

    # --- 6. 测验与实验结果生成 ---
    model.eval()
    with torch.no_grad():
        # 计算RAG相似度特征（FAISS）
        rag_features = []
        for emb in X_test.cpu().numpy():
            rag_features.append(get_max_similarities_faiss(emb, role_kb, num_roles))
        rag_features = np.array(rag_features)  # shape: (N, 4)
        # 拼接到BERT特征
        X_test_rag = torch.tensor(np.concatenate([X_test.cpu().numpy(), rag_features], axis=1), dtype=torch.float32).to(device)
        test_outputs = model(X_test_rag)
        _, y_pred = torch.max(test_outputs, 1)
        y_pred = y_pred.cpu().numpy()
        y_true = y_test.numpy()

    # 动态危险操作判别 + 角色预测判别
    print("正在进行动态危险操作判别与角色预测...")
    # 获取测试集对应的原始索引
    test_indices = X_test.cpu().numpy().shape[0]
    test_queries = df_l2.iloc[-test_indices:]['query'].values
    test_roles = df_l2.iloc[-test_indices:]['role'].values
    test_labels = labels_test

    final_pred = []
    rag_threshold = 0.7  # 可调阈值
    for i, (query, true_role, pred_role) in enumerate(zip(test_queries, test_roles, y_pred)):
        rag_sim = rag_features[i][true_role]
        # 1. RAG一致性判定
        if rag_sim < rag_threshold:
            final_pred.append(2)  # 判为伪装攻击
            continue
        # 2. 角色预测判定
        if pred_role != true_role:
            final_pred.append(2)
        else:
            final_pred.append(0)

    final_pred = np.array(final_pred)
    # 只统计预测为正常的准确率
    mask = (final_pred == 0)
    acc = np.mean(test_labels[mask] == 0) if np.sum(mask) > 0 else 0

    print(f"最终通过语句数: {np.sum(mask)} / {len(final_pred)}")
    print(f"最终准确率（通过语句与Label=0的重合度）: {acc:.4f}")

    # 绘制混淆矩阵（颜色改成绿色，与layer1隔开）
    plot_confusion_matrix(
        y_true, y_pred, output_dir,
        labels=[f'R{i}' for i in range(num_roles)],
        filename='confusion_matrix.png',
        cmap='Greens',
        title='Layer 2: Role Accountability Matrix'
    )

    # 保存分类报告 TXT
    write_classification_report(
        y_true, y_pred, output_dir,
        labels=[f'R{i}' for i in range(num_roles)],
        filename='layer2_report.txt',
        note="This model predicts which role 'usually' writes such SQL.\nIf True Role != Predicted Role, it might be a violation."
    )

    # --- 7. 生成详细推理日志 (前200条) ---
    detail_path = os.path.join(output_dir, 'role_probability_details.txt')
    print("正在记录前200条SQL的概率分布明细...")

    # 取前200条（或者更少）进行推理
    sample_limit = min(200, len(df_l2))
    sample_embs = torch.tensor(
        embedder.get_embeddings(df_l2['clean_query'].iloc[:sample_limit].values, batch_size=32)).to(device)
    rag_sample_features = []
    for emb in sample_embs.cpu().numpy():
        rag_sample_features.append(get_max_similarities_faiss(emb, role_kb, num_roles))
    rag_sample_features = np.array(rag_sample_features)
    sample_input = torch.tensor(np.concatenate([sample_embs.cpu().numpy(), rag_sample_features], axis=1), dtype=torch.float32).to(device)

    with torch.no_grad():
        probs = model(sample_input).cpu().numpy()  # shape (200, 4)
        _, preds = torch.max(torch.tensor(probs), 1)

    # 为前200条样本计算本层判定结果（0/2）
    pred_layer2 = []
    for i in range(sample_limit):
        query = df_l2.loc[i, 'query']
        true_role = df_l2.loc[i, 'role']
        pred_role = preds[i].item()
        rag_sim = rag_sample_features[i][true_role]
        if rag_sim < rag_threshold:
            pred_layer2.append(2)
        elif pred_role != true_role:
            pred_layer2.append(2)
        else:
            pred_layer2.append(0)
    actual_label = df_l2['Label'].iloc[:sample_limit].values

    write_detail_log(
        detail_path, df_l2, preds, probs, rag_sample_features, pred_layer2, actual_label, sample_limit=sample_limit
    )

    print(f"Layer 2 实验完成！请查看: {output_dir}")


if __name__ == "__main__":
    main_layer2()
