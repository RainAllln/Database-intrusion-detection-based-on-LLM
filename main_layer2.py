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
from src.detector_layer2 import Layer2Classifier
from src import SQLPreprocessor, SQLEmbedder


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

    # 提取 Embedding
    X_embeddings = embedder.get_embeddings(df_l2['clean_query'].values, batch_size=128)
    # y 标签在这里是 role，训练模型去识别这个语句属于哪个角色
    y_roles = torch.tensor(df_l2['role'].values).long()

    # --- 4. 划分数据集 ---
    X_train, X_test, y_train, y_test = train_test_split(
        X_embeddings, y_roles, test_size=0.2, random_state=42
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

    # --- 5. 模型训练 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Layer2Classifier(num_roles=num_roles).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)

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
    plt.figure()
    plt.plot(loss_history, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Layer 2 Training Loss')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'training_loss.png'))
    plt.close()

    # --- 6. 测验与实验结果生成 ---
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test.to(device))
        _, y_pred = torch.max(test_outputs, 1)
        y_pred = y_pred.cpu().numpy()
        y_true = y_test.numpy()

    # 绘制混淆矩阵（颜色改成绿色，与layer1隔开）
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    role_labels = [f'R{i}' for i in range(num_roles)]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=role_labels, yticklabels=role_labels)
    plt.title('Layer 2: Role Accountability Matrix')
    plt.ylabel('True Role (Who issued it)')
    plt.xlabel('Predicted Role (Who SHOULD issue it)')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

    # 保存分类报告 TXT
    report = classification_report(y_true, y_pred, target_names=role_labels)
    acc = accuracy_score(y_true, y_pred)

    report_path = os.path.join(output_dir, 'layer2_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=== SQL Detection Layer 2 (Role Perception) Report ===\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total L2 Samples (Train+Test): {len(df_l2)}\n")
        f.write(f"Test Set Accuracy: {acc:.4f}\n")
        f.write("Note: This model predicts which role 'usually' writes such SQL.\n")
        f.write("      If True Role != Predicted Role, it might be a violation.\n\n")
        f.write("--- Classification Report ---\n")
        f.write(report)

    # --- 7. 生成详细推理日志 (前100条) ---
    detail_path = os.path.join(output_dir, 'role_probability_details.txt')
    print("正在记录前100条SQL的概率分布明细...")

    # 取前100条（或者更少）进行推理
    sample_limit = min(100, len(df_l2))
    sample_embs = torch.tensor(
        embedder.get_embeddings(df_l2['clean_query'].iloc[:sample_limit].values, batch_size=32)).to(device)

    with torch.no_grad():
        probs = model(sample_embs).cpu().numpy()  # shape (100, 4)
        _, preds = torch.max(torch.tensor(probs), 1)

    with open(detail_path, 'w', encoding='utf-8') as f:
        # 表头
        headers = ["ID", "True_Role", "Pred_Role", "Result", "Confidence", "SQL_Snippet"]
        f.write(
            f"{headers[0]:<5} | {headers[1]:<10} | {headers[2]:<10} | {headers[3]:<8} | {headers[4]:<10} | {headers[5]}\n")
        f.write("-" * 120 + "\n")

        for i in range(sample_limit):
            query_clip = df_l2.loc[i, 'query'][:50].replace('\n', ' ') + "..."
            actual_role = df_l2.loc[i, 'role']
            pred_role = preds[i].item()

            # Confidence 是预测角色的概率值
            confidence = probs[i][pred_role]

            # 判断逻辑：如果 真实角色 != 预测角色，这可能就是伪造的 "越权(Label 2)"
            is_match = (actual_role == pred_role)
            res_mark = "MATCH" if is_match else "MISMATCH"

            f.write(
                f"{i:<5} | {actual_role:<10} | {pred_role:<10} | {res_mark:<8} | {confidence:.4f}     | {query_clip}\n")

    print(f"Layer 2 实验完成！请查看: {output_dir}")


if __name__ == "__main__":
    main_layer2()