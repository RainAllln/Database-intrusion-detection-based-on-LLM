import os
import torch
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from src import SQLPreprocessor, SQLEmbedder, Layer2Classifier
from src.rag import (
    build_role_knowledge_base_faiss_l2,
    get_top1_l2_distances_faiss,
)
from src.utils import (
    plot_confusion_matrix,
    plot_loss_curve,
    write_detail_log,
    plot_rag_similarity_distribution,
    write_experiment_report,
)

def main_layer2():
    # === 1. 配置 ===
    data_type = "complex"
    data_path = 'G:/graduate_pro/SqlDetection/data/custom/complex_dataset_v3.csv'
    extractor_type = "distilbert"
    feature_model_name = "distilbert-base-uncased"
    layer2_model_name = "distilbert_clf"
    layer2_train_hparams = {
        "num_train_epochs": 3,
        "per_device_train_batch_size": 16,
        "per_device_eval_batch_size": 32,
        "learning_rate": 5e-5,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "logging_steps": 50,
    }
    num_roles = 8
    rag_threshold = 0.7

    # 2. 加载数据 & 预处理
    print(f"正在读取数据集: {data_path}")
    try:
        df = pd.read_csv(data_path, encoding='utf-8')
    except (UnicodeError, FileNotFoundError, Exception):
        try:
            df = pd.read_csv(data_path, encoding='gbk')
        except:
            df = pd.read_csv(data_path, encoding='latin-1')

    # 统一 Label 类型
    df['Label'] = pd.to_numeric(df['Label'], errors='coerce').fillna(0).astype(int)

    print("筛选数据: 过滤掉 Label 1 (SQL注入)，只保留正常(0)和越权(2)...")
    df_l2 = df[df['Label'] != 1].reset_index(drop=True)
    print(f"第二层有效样本数: {len(df_l2)}")

    # SQL 预处理
    preprocessor = SQLPreprocessor()
    df_l2['ast_query'] = df_l2['query'].astype(str).apply(preprocessor.normalize_and_flatten)

    # 3. 建立输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_folder = f"exp2_{data_type}_{layer2_model_name}_{timestamp}"
    output_dir = os.path.join("notebooks", exp_folder)
    os.makedirs(output_dir, exist_ok=True)
    print(f"第二层实验结果将保存至: {output_dir}")

    # 4. 特征提取（DistilBERT）
    embedder = SQLEmbedder(
        extractor_type=extractor_type,
        model_name=feature_model_name,
    )
    batch_size = 128 if torch.cuda.is_available() else 32
    print("开始提取 AST SQL 的 DistilBERT 向量...")
    X_embeddings = embedder.get_embeddings(df_l2['ast_query'].values, batch_size=batch_size)

    # 角色标签编码为 [0, num_roles-1]
    le = LabelEncoder()
    y_roles = le.fit_transform(df_l2['role'].values)
    labels_all = df_l2['Label'].values

    # 5. 划分训练 / 测试集
    indices = np.arange(len(df_l2))
    train_idx, test_idx, _, _ = train_test_split(
        indices, indices, test_size=0.2, random_state=42
    )
    X_train = X_embeddings[train_idx]
    X_test = X_embeddings[test_idx]
    y_train = y_roles[train_idx]
    y_test = y_roles[test_idx]
    labels_train = labels_all[train_idx]
    labels_test = labels_all[test_idx]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 、6. 构建 RAG 角色知识库
    print("正在用FAISS构建RAG角色知识库（L2距离）...")
    role_kb = build_role_knowledge_base_faiss_l2(
        X_train, y_train, labels_train, num_roles
    )

    # 7. 训练集 RAG 特征 + 拼接
    rag_features_train = []
    for emb in X_train:
        rag_features_train.append(
            get_top1_l2_distances_faiss(emb, role_kb, num_roles)
        )
    rag_features_train = np.array(rag_features_train)

    # 根据模型类型组织训练输入
    if layer2_model_name == "distilbert_clf":
        # 对 DistilBERT 分类器：使用文本 + RAG 特征
        texts_train = df_l2.loc[train_idx, 'ast_query'].astype(str).tolist()
        X_train_final = {
            "texts": texts_train,
            "rag_features": rag_features_train.astype(np.float32),
        }
        y_train_np = y_train.astype(np.int64)
        input_dim = None
    else:
        # embedding + RAG 数值特征
        X_train_final = np.concatenate([X_train, rag_features_train], axis=1).astype(np.float32)
        y_train_np = y_train.astype(np.int64)
        input_dim = X_train_final.shape[1]

    # 8. 训练第二层分类器
    layer2_clf = Layer2Classifier(
        model_name=layer2_model_name,
        input_dim=input_dim if input_dim is not None else 0,
        num_roles=num_roles,
        device=device,
    )

    print(f"正在训练角色行为模型 (Model: {layer2_model_name}, Device: {device})...")
    loss_history = layer2_clf.train(
        X_train_final,
        y_train_np,
        training_hparams=layer2_train_hparams,
        output_dir=output_dir,
    )

    # 9. 绘制 loss 曲线
    try:
        if loss_history is not None and len(loss_history) > 0:
            plot_loss_curve(
                loss_history,
                output_dir,
                filename='layer2_training_loss.png',
                title='Layer2 Training Loss (per batch)'
            )
    except Exception as e:
        print(f"绘制训练损失曲线时出错: {e}")

    # 10. 测试集 RAG 特征 + 角色预测 + RAG 阈值判定
    print("正在进行静态RAG判别与角色预测...")
    rag_features_test = []
    for emb in X_test:
        rag_features_test.append(
            get_top1_l2_distances_faiss(emb, role_kb, num_roles)
        )
    rag_features_test = np.array(rag_features_test)

    if layer2_model_name == "distilbert_clf":
        texts_test = df_l2.loc[test_idx, 'ast_query'].astype(str).tolist()
        X_test_final = {
            "texts": texts_test,
            "rag_features": rag_features_test.astype(np.float32),
        }
    else:
        X_test_final = np.concatenate([X_test, rag_features_test], axis=1).astype(np.float32)

    # 角色分类器预测
    y_pred_idx, probs = layer2_clf.predict(X_test_final)
    y_pred_roles = le.inverse_transform(y_pred_idx)
    y_true_roles = le.inverse_transform(y_test)

    # 基于 RAG Top-1 距离 + 角色是否匹配 的最终 0/2 判定
    final_pred = []
    for i, pred_role in enumerate(y_pred_idx):
        # rag_features_* 的第 pred_role 维即为该预测角色的 L2 距离
        dist = float(rag_features_test[i][int(pred_role)])
        # 若角色预测正确且距离小于阈值 => 判为正常(0)，否则判为越权(2)
        if (pred_role == y_test[i]) and (dist < rag_threshold):
            final_pred.append(0)
        else:
            final_pred.append(2)
    final_pred = np.array(final_pred)

    # 11. 可视化
    test_labels_02 = labels_test  # 仅包含 0 / 2
    overall_acc = accuracy_score(test_labels_02, final_pred)

    print("\n=== 第二层 (RAG+角色模型) 评估报告 ===")
    print(f"使用模型: {layer2_model_name}")
    print(f"最终准确率（0 vs 2）: {overall_acc:.4f}")
    print(f"标签分布: {dict(zip(*np.unique(test_labels_02, return_counts=True)))}")

    print("\n--- 角色分类报告（编码空间） ---")
    print(classification_report(
        y_test, y_pred_idx,
        target_names=[f'R{i}' for i in range(num_roles)],
        zero_division=0
    ))

    # 0 / 2 最终决策混淆矩阵
    plot_confusion_matrix(
        test_labels_02, final_pred, output_dir,
        labels=[0, 2],
        filename='confusion_matrix_final.png',
        cmap='Blues',
        title='Layer 2: Final Decision Matrix (0=Normal, 2=Impersonation)'
    )
    # 角色分类混淆矩阵
    plot_confusion_matrix(
        y_test, y_pred_idx, output_dir,
        labels=[f'R{i}' for i in range(num_roles)],
        filename='confusion_matrix_role.png',
        cmap='Greens',
        title=f'Layer 2 ({layer2_model_name}) Role Classification Matrix'
    )

    # 12. RAG 相似度分布
    try:
        rag_sim_distributions = {r: [] for r in range(num_roles)}
        for i, r in enumerate(y_train_np):
            dist = float(rag_features_train[i][int(r)])
            sim = 1.0 / (1.0 + dist)
            rag_sim_distributions[int(r)].append(sim)
        plot_rag_similarity_distribution(
            rag_sim_distributions,
            output_dir,
            filename='rag_similarity_distribution.png'
        )
    except Exception as e:
        print(f"绘制 RAG 相似度分布时出错: {e}")

    # 13. 详细日志
    detail_path = os.path.join(output_dir, 'role_probability_details.txt')
    print("正在记录前200条SQL的概率分布明细...")

    sample_limit = min(200, len(df_l2))
    sample_embs_np = X_embeddings[:sample_limit]

    rag_sample_features = []
    for emb in sample_embs_np:
        rag_sample_features.append(
            get_top1_l2_distances_faiss(emb, role_kb, num_roles)
        )
    rag_sample_features = np.array(rag_sample_features)

    if layer2_model_name == "distilbert_clf":
        sample_texts = df_l2['ast_query'].iloc[:sample_limit].astype(str).tolist()
        sample_input_np = {
            "texts": sample_texts,
            "rag_features": rag_sample_features.astype(np.float32),
        }
    else:
        sample_input_np = np.concatenate([sample_embs_np, rag_sample_features], axis=1).astype(np.float32)

    sample_preds_idx, sample_probs = layer2_clf.predict(sample_input_np)

    pred_layer2 = []
    for i in range(sample_limit):
        true_role_idx = le.transform([df_l2.loc[i, 'role']])[0]
        pred_role_idx = int(sample_preds_idx[i])
        dist = float(rag_sample_features[i][pred_role_idx])
        if (pred_role_idx == true_role_idx) and (dist < rag_threshold):
            pred_layer2.append(0)
        else:
            pred_layer2.append(2)
    actual_label = df_l2['Label'].iloc[:sample_limit].values

    write_detail_log(
        detail_path, df_l2,
        torch.tensor(sample_preds_idx),
        sample_probs,
        rag_sample_features,
        pred_layer2,
        actual_label,
        sample_limit=sample_limit
    )

    # 14. 写入txt文件
    report_path = os.path.join(output_dir, 'experiment_report.txt')
    title = "SQL Detection Layer 2 Experiment Report"

    try:
        hyper_str = layer2_clf.get_hyperparams_str(
            training_hparams=layer2_train_hparams,
            extra_params={"rag_threshold": rag_threshold, "num_roles": num_roles},
        )
    except Exception:
        hyper_str = f"Layer2: {layer2_model_name} (no hyperparams_str)"

    model_paras = (
        f"DataSet: {data_path}\n"
        f"Model: {layer2_model_name}+RAG\n"
        f"Embedding: {extractor_type} ({feature_model_name}) on AST normalized SQL\n"
        f"{hyper_str}\n"
        f"训练集样本数: {len(X_train)}\n"
        f"测试集样本数: {len(X_test)}\n"
        f"角色数: {num_roles}\n"
    )
    note = (
        f"最终准确率（0 vs 2）: {overall_acc:.4f}\n"
    )
    content = (
        "\n--- 角色分类报告（编码空间） ---\n"
        f"{classification_report(y_test, y_pred_idx, target_names=[f'R{i}' for i in range(num_roles)], zero_division=0)}\n"
    )

    write_experiment_report(report_path, title, model_paras, note, content)
    print(f"Layer 2 简化实验完成！请查看: {output_dir}")

if __name__ == "__main__":
    main_layer2()
