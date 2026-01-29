import os
from datetime import datetime
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score, confusion_matrix, roc_curve
from src import SQLPreprocessor, SQLEmbedder
from src.detector_layer1 import Layer1Detector


def plot_metrics(y_true, y_pred, scores, output_dir):
    """绘制并保存评估图表"""
    # 1. 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    save_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(save_path)
    print(f"图表已保存: {save_path}")
    plt.close()

    # 2. ROC 曲线
    fpr, tpr, _ = roc_curve(y_true, -scores) # 注意 scores 取负
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc_score(y_true, -scores):.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    
    save_path = os.path.join(output_dir, 'roc_curve.png')
    plt.savefig(save_path)
    print(f"图表已保存: {save_path}")
    plt.close()

    # 3. 分数分布图
    plt.figure(figsize=(8, 5))
    df_res = pd.DataFrame({'Score': scores, 'Label': y_true})
    sns.histplot(data=df_res, x='Score', hue='Label', element="step", stat="density", common_norm=False, bins=30)
    plt.title('Anomaly Score Distribution (Layer 1)')
    plt.xlabel('Anomaly Score (Lower is more anomalous)')
    
    save_path = os.path.join(output_dir, 'score_dist.png')
    plt.savefig(save_path)
    print(f"图表已保存: {save_path}")
    plt.close()


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
    
    # 确保 Label 是数值型 (2：伪装攻击， 1：注入攻击，0：正常)
    df['Label'] = pd.to_numeric(df['Label'], errors='coerce').fillna(0).astype(int)

    # 2. 标签逻辑调整 (针对三分类)
    # 第一层模型：只管检测是否为 SQL 注入 (Label 1)
    # 将 Label 0 和 2 都视为“语法层面的正常” (即非注入)
    y_for_layer1 = df['Label'].apply(lambda x: 1 if x == 1 else 0).values

    # 3. 特征提取
    embedder = SQLEmbedder()
    run_batch_size = 128 if torch.cuda.is_available() else 32
    embeddings = embedder.get_embeddings(df['clean_query'].values, batch_size=run_batch_size)

    # 4. 划分训练集
    X_train, X_test, y_train, y_test = train_test_split(embeddings, y_for_layer1, test_size=0.2, random_state=42)

    # 第一层训练策略：仅使用 Label 为 0 和 2 的样本进行训练（因为它们语法正常）
    # 这样孤立森林会把 Label 1 这种结构怪异的语句识别为离群点
    X_train_normal = X_train[y_train == 0]

    contamination_rate = 0.05  # 假设预期误报率为 5%

    l1_detector = Layer1Detector(contamination_rate)

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
    l1_detector = Layer1Detector(contamination=contamination_rate)
    l1_detector.train(X_train_normal)

    # 5. 在测试集上评估
    print("正在评估第一层模型...")
    preds, scores = l1_detector.detect(X_test)

    # 孤立森林输出: -1(异常/攻击), 1(正常)
    # 真实标签: 1(攻击), 0(正常)
    y_pred_mapped = [1 if p == -1 else 0 for p in preds]

    # 绘制图表
    plot_metrics(y_test, y_pred_mapped, scores, output_dir)

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
    # 获取 Sklearn 模型的参数
    model_params = l1_detector.model.get_params()
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=== SQL Detection Layer 1 Experiment Report ===\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("--- Data Info ---\n")
        f.write(f"Total Samples: {len(X)}\n")
        f.write(f"Training Strategy: Semi-Supervised (Normal Samples Only)\n")
        f.write(f"Training Samples Used: {len(X_train_normal)}\n")
        f.write(f"Test Samples: {len(X_test)}\n\n")
        
        f.write("--- Model Parameters (Isolation Forest) ---\n")
        for k, v in model_params.items():
            f.write(f"{k}: {v}\n")
        f.write("\n")
        
        f.write("--- Evaluation Metrics ---\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"ROC-AUC Score: {roc_score if isinstance(roc_score, str) else f'{roc_score:.4f}'}\n\n")
        
        f.write("--- Classification Report ---\n")
        f.write(cls_report)
        
    print(f"实验参数与结果已写入: {report_path}")

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

