import os
from datetime import datetime
import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from src import SQLPreprocessor, SQLEmbedder, Layer1Detector, Layer2Classifier
from src.rag import build_role_knowledge_base_faiss_l2, get_top1_l2_distances_faiss
from src.utils import (
    plot_confusion_matrix,
    plot_score_distribution,
    plot_roc_curve,
    plot_rag_similarity_distribution,
    plot_loss_curve
)

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
    # 预处理
    df['ast_query'] = df['query'].apply(preprocessor.normalize_and_flatten)
    df['Label'] = pd.to_numeric(df['Label'], errors='coerce').fillna(0).astype(int)

    # 2. 特征提取（DistilBERT）
    embedder = SQLEmbedder()
    batch_size = 128 if torch.cuda.is_available() else 32
    embeddings = embedder.get_embeddings(df['ast_query'].values, batch_size=batch_size)

    # --- 训练相关参数 & RAG 参数 ---
    mlp_train_hparams = {
        "lr": 0.001,
        "epochs": 15,
        "batch_size": 32,
    }
    rag_params = {
        "num_roles": 4,
        "rag_threshold": 0.7,
        "faiss_metric": "L2",
    }
    num_roles = rag_params["num_roles"]
    rag_threshold = rag_params["rag_threshold"]

    # 3. 划分训练集和测试集（全流程统一划分）
    train_idx, test_idx = train_test_split(np.arange(len(df)), test_size=0.2, random_state=42)
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    train_embeddings = embeddings[train_idx]
    test_embeddings = embeddings[test_idx]

    # 4. 第一层训练（LOF，仅正常样本）
    y_train_layer1 = train_df['Label'].apply(lambda x: 1 if x == 1 else 0).values
    X_train_normal = train_embeddings[y_train_layer1 == 0]
    # 使用 LOF 模型
    l1_detector = Layer1Detector(model_name="lof")
    l1_detector.train(X_train_normal)

    # 5. 第二层训练（MLP+RAG角色分类）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 训练集中过滤掉注入(Label!=1)，用于角色分类与KB构建
    train_roles_mask = train_df['Label'] != 1
    X_train2 = train_embeddings[train_roles_mask]
    y_train2 = train_df.loc[train_roles_mask, 'role'].values
    labels_train2 = train_df.loc[train_roles_mask, 'Label'].values

    # 构建FAISS角色知识库（仅Label==0）——调用 rag.py
    role_kb = build_role_knowledge_base_faiss_l2(
        X_train2, y_train2, labels_train2, num_roles
    )

    # 训练时计算RAG Top-1 L2距离并拼接到embedding —— 调用 rag.py
    rag_features_train = []
    for emb in X_train2:
        rag_features_train.append(
            get_top1_l2_distances_faiss(emb, role_kb, num_roles)
        )
    rag_features_train = np.array(rag_features_train)
    X_train2_rag = np.concatenate([X_train2, rag_features_train], axis=1)

    X_train2_tensor = torch.tensor(X_train2_rag, dtype=torch.float32).to(device)
    y_train2_tensor = torch.tensor(y_train2, dtype=torch.long).to(device)

    model = Layer2Classifier(model_name="mlp", num_roles=num_roles).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=mlp_train_hparams["lr"])
    criterion = torch.nn.CrossEntropyLoss()
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train2_tensor, y_train2_tensor),
        batch_size=mlp_train_hparams["batch_size"],
        shuffle=True
    )
    model.train()

    loss_history = []
    for epoch in range(mlp_train_hparams["epochs"]):
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            # 记录每个 batch 的 loss（可用 plot_loss_curve 可视化）
            loss_history.append(loss.item())

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

    # 计算RAG特征，拼接后推理 —— 调用 rag.py
    rag_features_test = []
    for emb in l2_embeddings:
        rag_features_test.append(
            get_top1_l2_distances_faiss(emb, role_kb, num_roles)
        )
    rag_features_test = np.array(rag_features_test)
    l2_input = np.concatenate([l2_embeddings, rag_features_test], axis=1)

    model.eval()
    with torch.no_grad():
        l2_tensor = torch.tensor(l2_input, dtype=torch.float32).to(device)
        outputs = model(l2_tensor)
        _, l2_roles_pred = torch.max(outputs, 1)
        l2_roles_pred = l2_roles_pred.cpu().numpy()

    # 基于RAG距离与角色匹配的最终判定
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

    # 说明：计算分层指标以便诊断
    overall_acc = accuracy_score(true_label, framework_pred)

    # 标签分布与第一层统计
    unique, counts = np.unique(true_label, return_counts=True)
    label_counts = dict(zip(unique, counts))
    num_first_layer_injected = np.sum(l1_result == 1)
    num_enter_l2 = np.sum(l1_result == 0)

    # 第一层（注入检测）指标
    # 真实注入样本掩码
    true_injection_mask = (true_label == 1).astype(int)
    # l1_result: 1 表示被判为注入
    first_layer_overall_acc = accuracy_score(true_injection_mask, l1_result)
    # 对注入样本的召回（在所有真实注入样本中被第一层正确标注为注入的比例）
    if np.sum(true_injection_mask) > 0:
        first_layer_injection_recall = np.sum((l1_result == 1) & (true_injection_mask == 1)) / np.sum(true_injection_mask)
    else:
        first_layer_injection_recall = 0.0

    # 第二层（伪装 vs 正常）指标：仅在第一层通过的样本上计算
    if num_enter_l2 > 0:
        # l2_labels_true: ground truth (0 or 2)
        # final_l2_pred: final labels (0 or 2) for those samples
        second_layer_overall_acc = accuracy_score(l2_labels_true, final_l2_pred)
        # 对伪装(Label==2)样本的召回（在进入第二层的真实伪装样本中被判为2的比例）
        mask_real_impersonation = (l2_labels_true == 2)
        if np.sum(mask_real_impersonation) > 0:
            second_layer_impersonation_recall = np.sum((np.array(final_l2_pred) == 2) & mask_real_impersonation) / np.sum(mask_real_impersonation)
        else:
            second_layer_impersonation_recall = 0.0
    else:
        second_layer_overall_acc = 0.0
        second_layer_impersonation_recall = 0.0

    print(f"\n=== 分层检测统计 ===")
    print(f"测试集总数: {len(test_df)}")
    print(f"真实标签分布: {label_counts}")
    print(f"第一层判定为注入（直接标为1）的样本数: {num_first_layer_injected}")
    print(f"第一层通过进入第二层的样本数: {num_enter_l2}")

    print("\n-- 第一层 (Injection Detection) --")
    print(f"第一层在整体测试集上的准确率: {first_layer_overall_acc:.4f}")
    print(f"第一层对真实注入样本的召回 (Recall for Label=1): {first_layer_injection_recall:.4f}")

    print("\n-- 第二层 (Impersonation Detection on passed samples) --")
    print(f"第二层在进入样本上的分类准确率 (0 vs 2): {second_layer_overall_acc:.4f}")
    print(f"第二层对真实伪装样本的召回 (Recall for Label=2 within passed samples): {second_layer_impersonation_recall:.4f}")

    print(f"\n=== 整体框架准确率 (overall): {overall_acc:.4f} ===")
    print(f"框架判定为各类分布: {dict(zip(*np.unique(framework_pred, return_counts=True)))}")

    print("\n=== 详细分类报告（框架输出标签与真实标签） ===")
    print(classification_report(true_label, framework_pred, target_names=['正常', '注入', '伪装'], zero_division=0))

    # 构造并写入实验报告（确保所有变量已初始化，避免 UnboundLocalError）
    timestamp_report = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir_report = os.path.join("notebooks", f"exp_main_{timestamp_report}")
    os.makedirs(output_dir_report, exist_ok=True)
    report_path = os.path.join(output_dir_report, 'experiment_report.txt')



    # 1) 混淆矩阵（整体框架）
    plot_confusion_matrix(true_label, framework_pred, output_dir_report, labels=['0', '1', '2'],
                          filename='confusion_matrix.png', title='Framework Confusion Matrix')

    # 2) 第一层（孤立森林）分数分布与 ROC（以检测注入 Label==1 为正类）
    try:
        true_injection_mask = (true_label == 1).astype(int)
        plot_score_distribution(true_injection_mask, scores, output_dir_report,
                                filename='layer1_score_distribution.png',
                                title='Layer1 Anomaly Score Distribution')
        plot_roc_curve(true_injection_mask, scores, output_dir_report, filename='layer1_roc_curve.png')
    except Exception as e:
        print(f"绘制第一层分布/ROC 时出错: {e}")

    # 3) RAG 相似度分布（将 L2 距离转换为相似度后绘图）
    try:
        # rag_features_train: shape (N_train2, num_roles)
        rag_sim_distributions = {r: [] for r in range(num_roles)}
        for i, r in enumerate(y_train2):
            dist = float(rag_features_train[i][int(r)])
            sim = 1.0 / (1.0 + dist)  # 将距离转换为相似度（0..1]
            rag_sim_distributions[int(r)].append(sim)
        plot_rag_similarity_distribution(rag_sim_distributions, output_dir_report,
                                         filename='rag_similarity_distribution.png')
    except Exception as e:
        print(f"绘制 RAG 相似度分布时出错: {e}")

    # 4) 训练损失曲线（若有记录）
    try:
        if len(loss_history) > 0:
            plot_loss_curve(loss_history, output_dir_report, filename='layer2_training_loss.png',
                            title='Layer2 Training Loss (per batch)')
    except Exception as e:
        print(f"绘制训练损失曲线时出错: {e}")

    print(f"所有图表已保存到: {output_dir_report}")

    title = "SQL Detection Framework Experiment Report"

    # --- 由模型自己返回超参数字符串，main 只做拼接 ---
    try:
        l1_hyperparams_str = l1_detector.get_hyperparams_str()
    except Exception as e:
        print(f"获取第一层超参数字符串失败: {e}")
        l1_hyperparams_str = "Layer1: (failed to get hyperparams_str)"

    try:
        # 第二层模型需要接受训练超参和 RAG 超参，内部自己决定如何展示
        l2_hyperparams_str = model.get_hyperparams_str(
            training_hparams=mlp_train_hparams,
            rag_params=rag_params,
        )
    except Exception as e:
        print(f"获取第二层超参数字符串失败: {e}")
        l2_hyperparams_str = "Layer2: (failed to get hyperparams_str)"

    # --- 最终写入 txt 的 model_paras ---
    model_paras = (
        f"{l1_hyperparams_str}\n\n"
        f"{l2_hyperparams_str}\n\n"
        f"Train samples: {len(train_embeddings)}\n"
        f"Test samples: {len(test_embeddings)}\n"
    )

    note = (
        f"First Layer (Injection) - overall accuracy: {first_layer_overall_acc:.4f}\n"
        f"First Layer (Injection) - recall for Label=1: {first_layer_injection_recall:.4f}\n"
        f"Second Layer (Impersonation) - accuracy on passed samples: {second_layer_overall_acc:.4f}\n"
        f"Second Layer (Impersonation) - recall for Label=2 (within passed): {second_layer_impersonation_recall:.4f}\n"
        f"Overall Framework Accuracy: {overall_acc:.4f}\n"
    )
    content = (
        "\n--- Classification Report ---\n" +
        classification_report(true_label, framework_pred, target_names=['正常', '注入', '伪装'], zero_division=0)
    )
    from src.utils import write_experiment_report
    write_experiment_report(report_path, title, model_paras, note, content)

if __name__ == "__main__":
    main()

