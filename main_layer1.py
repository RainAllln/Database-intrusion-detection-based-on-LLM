import os
from datetime import datetime
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score, confusion_matrix, roc_curve
from src import SQLPreprocessor, SQLEmbedder, Layer1Detector
from src.utils import plot_confusion_matrix, plot_roc_curve, plot_score_distribution, write_experiment_report


def main():
    # 1. 加载与预处理
    # 尝试使用 utf-16 编码读取
    try:
        # df = pd.read_csv('data/raw/sqlInjection/sqli.csv', encoding='utf-16')
        df = pd.read_csv('data/custom/custom_dataset.csv', encoding='utf-16') # 读取我生成的数据集
    except UnicodeError:
        # 如果不是 utf-16，尝试 latin-1 读取
        # df = pd.read_csv('data/raw/sqlInjection/sqli.csv', encoding='latin-1')
        df = pd.read_csv('data/custom/custom_dataset.csv', encoding='latin-1')  # 读取我生成的数据集

    preprocessor = SQLPreprocessor()
    # 修改列名为 'query' 以匹配 csv 文件头
    df['clean_query'] = df['query'].apply(preprocessor.normalize)
    # 新增：AST展平SQL
    df['ast_query'] = df['query'].apply(lambda x: " ".join(preprocessor.get_ast_sequence(x)))

    # 确保 Label 是数值型 (2：伪装攻击， 1：注入攻击，0：正常)
    df['Label'] = pd.to_numeric(df['Label'], errors='coerce').fillna(0).astype(int)

    # 2. 标签逻辑调整 (针对三分类)
    # 第一层模型：只管检测是否为 SQL 注入 (Label 1)
    # 将 Label 0 和 2 都视为“语法层面的正常” (即非注入)
    y_for_layer1 = df['Label'].apply(lambda x: 1 if x == 1 else 0).values

    # 3. 特征提取
    embedder = SQLEmbedder()
    run_batch_size = 128 if torch.cuda.is_available() else 32
    # 可选：用AST特征或normalize特征
    # embeddings = embedder.get_embeddings(df['clean_query'].values, batch_size=run_batch_size)
    embeddings = embedder.get_embeddings(df['ast_query'].values, batch_size=run_batch_size)

    # 4. 划分训练集
    X_train, X_test, y_train, y_test = train_test_split(embeddings, y_for_layer1, test_size=0.2, random_state=42)

    # 第一层训练策略：仅使用 Label 为 0 和 2 的样本进行训练（因为它们语法正常）
    # 这样孤立森林会把 Label 1 这种结构怪异的语句识别为离群点
    X_train_normal = X_train[y_train == 0]

    contamination_rate = 0.05  # 假设预期误报率为 5%

    l1_detector = Layer1Detector(model_name="isolation_forest")

    l1_detector.train(X_train_normal)

    # --- 创建实验文件夹 ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 文件夹命名：exp1_cont[污染率]_[时间戳]，表示第一层模型的实验
    exp_folder = f"exp1_cont{contamination_rate:.4f}_{timestamp}"
    output_dir = os.path.join("notebooks", exp_folder)
    os.makedirs(output_dir, exist_ok=True)
    print(f"本次实验结果将保存至: {output_dir}")
    # -------------------
    
    print(f"初始化孤立森林，仅使用正常样本训练，污染率参数(预期误报): {contamination_rate:.4f}")
    print(f"训练样本数 (Normal Only): {len(X_train_normal)}")
    l1_detector = Layer1Detector(model_name="isolation_forest")
    l1_detector.train(X_train_normal)

    # 5. 在测试集上评估
    print("正在评估第一层模型...")
    preds, scores = l1_detector.detect(X_test)

    # 孤立森林输出: -1(异常/攻击), 1(正常)
    # 真实标签: 1(攻击), 0(正常)
    y_pred_mapped = [1 if p == -1 else 0 for p in preds]

    # 绘制图表
    plot_confusion_matrix(y_test, y_pred_mapped, output_dir, labels=['Normal', 'Attack'])
    plot_roc_curve(y_test, scores, output_dir)
    plot_score_distribution(y_test, scores, output_dir)

    # --- 计算指标 ---
    acc = accuracy_score(y_test, y_pred_mapped)
    f1 = f1_score(y_test, y_pred_mapped)
    try:
        roc_score = roc_auc_score(y_test, -scores)
    except Exception as e:
        roc_score = "N/A"
        
    cls_report = classification_report(y_test, y_pred_mapped, target_names=['Normal', 'Attack'])

    # --- 打印在控制台 ---
    print("\n=== 第一层 (Isolation Forest) 评估报告 ===")
    print(f"准确率 (Accuracy): {acc:.4f}")
    print(f"F1 分数 (F1 Score): {f1:.4f}")
    print(f"ROC-AUC Score: {roc_score if isinstance(roc_score, str) else f'{roc_score:.4f}'}")
    print("\n详细分类报告:")
    print(cls_report)
    
    # --- 保存实验参数与结果到 TXT ---
    report_path = os.path.join(output_dir, 'experiment_report.txt')
    report_title = "SQL Detection Layer 1 Experiment Report"
    report_note = (
        f"Total Samples: {len(X_train) + len(X_test)}\n"
        "Training Strategy: Semi-Supervised (Normal Samples Only)\n"
        f"Training Samples Used: {len(X_train)}\n"
        f"Test Samples: {len(X_test)}\n\n"
    )
    report_params = (
        f"preprocessor: AST\n"
        f"Model: Isolation Forest\n"
        f"Contamination Rate: {contamination_rate:.4f}\n"
        "Feature Type: AST Flattened Sequences\n"
    )
    report_content = (
        "--- Evaluation Metrics ---\n"
        f"Accuracy: {acc:.4f}\n"
        f"F1 Score: {f1:.4f}\n"
        f"ROC-AUC Score: {roc_score if isinstance(roc_score, str) else f'{roc_score:.4f}'}\n\n"
        "--- Classification Report ---\n"
        f"{cls_report}"
    )
    write_experiment_report(report_path,report_title,report_params,report_note, report_content)

    # 模拟单个新查询测试
    new_query = "SELECT * FROM users WHERE id = 1 OR 1=1 --"
    clean_new = preprocessor.normalize(new_query)
    new_emb = embedder.get_embeddings([clean_new])

    pred, score = l1_detector.detect(new_emb)

    if pred[0] == -1:
        print(f"第一层检测到外部注入攻击！分数: {score[0]:.4f}")
    else:
        print("第一层判定正常，进入第二层角色行为比对... ")
        # 这里后续调用 Layer2Classifier 逻辑


if __name__ == "__main__":
    main()

