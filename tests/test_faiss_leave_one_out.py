import os
import sys
import numpy as np
import pandas as pd
import faiss
import matplotlib.pyplot as plt
import seaborn as sns

# 获取项目根目录（SqlDetection）
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from src.feature import SQLEmbedder
from src.preprocess import SQLPreprocessor

def build_role_kb_leave_one_out(X, y, labels, num_roles):
    """
    针对每个角色，构建正常SQL的向量库（Label==0），返回每个角色的向量集合
    """
    kb = {}
    for r in range(num_roles):
        idx = (labels == 0) & (y == r)
        vecs = X[idx]
        kb[r] = vecs.astype('float32')
    return kb

def leave_one_out_similarities(vecs):
    """
    对于每个向量，计算其与库中除自己外的最相似（最小L2距离）向量的相似度
    """
    n, dim = vecs.shape
    if n <= 1:
        return np.array([1.0])  # 只有一个样本，返回最大相似度
    index = faiss.IndexFlatL2(dim)
    index.add(vecs)
    D, I = index.search(vecs, 2)  # 每个向量找最近的两个（第一个是自己）
    # D[:, 1] 是每个向量与库中除自己外的最近距离
    sims = np.exp(-D[:, 1])  # 距离转相似度
    return sims

def cross_role_similarity(role_vecs, kb, num_roles):
    """
    计算每个角色的SQL在其它角色知识库中的最大相似度分布
    返回: {role_i: {role_j: [sim_list]}}
    """
    cross_sim = {}
    for i in range(num_roles):
        cross_sim[i] = {}
        vecs_i = role_vecs[i]
        for j in range(num_roles):
            if i == j or len(kb[j]) == 0 or len(vecs_i) == 0:
                cross_sim[i][j] = np.array([])
                continue
            index = faiss.IndexFlatL2(kb[j].shape[1])
            index.add(kb[j])
            D, I = index.search(vecs_i, 1)  # 每个i角色SQL在j角色知识库里找最相似
            sims = np.exp(-D[:, 0])
            cross_sim[i][j] = sims
    return cross_sim

def plot_leave_one_out_distribution(sim_dict, output_dir, filename='faiss_leave_one_out.png'):
    plt.figure(figsize=(10, 6))
    for r, sims in sim_dict.items():
        sns.histplot(sims, bins=30, kde=True, label=f'Role {r}', stat='density', alpha=0.5)
    plt.xlabel('Leave-One-Out Max Similarity')
    plt.ylabel('Density')
    plt.title('Leave-One-Out Similarity Distribution for Each Role')
    plt.legend()
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path)
    plt.close()
    print(f"留一法相似度分布图已保存: {save_path}")

def plot_cross_role_distribution(cross_sim, output_dir):
    for i in cross_sim:
        plt.figure(figsize=(10, 6))
        for j in cross_sim[i]:
            sims = cross_sim[i][j]
            if len(sims) == 0:
                continue
            sns.histplot(sims, bins=30, kde=True, label=f'In Role {j} KB', stat='density', alpha=0.5)
        plt.xlabel('Max Similarity')
        plt.ylabel('Density')
        plt.title(f'Role {i} SQL in Other Roles KB Similarity')
        plt.legend()
        save_path = os.path.join(output_dir, f'cross_role_sim_role{i}.png')
        plt.savefig(save_path)
        plt.close()
        print(f"角色{i}在其它角色知识库的相似度分布图已保存: {save_path}")

def main():
    # 数据集路径修正
    data_path = os.path.join(project_root, 'data', 'custom', 'custom_dataset.csv')
    output_dir = os.path.join(project_root, 'tests', 'faiss_leave_one_out_results')
    os.makedirs(output_dir, exist_ok=True)

    print(f"读取数据: {data_path}")
    df = pd.read_csv(data_path, encoding='utf-8')
    num_roles = 4

    # 只取正常SQL
    mask = (df['Label'] == 0)
    df = df[mask].reset_index(drop=True)
    y = df['role'].values
    labels = df['Label'].values

    # 提取DistilBERT特征向量
    print("正在提取DistilBERT特征向量...")
    embedder = SQLEmbedder()
    # 可选：用AST展平或normalize
    preprocessor = SQLPreprocessor()
    df['ast_query'] = df['query'].astype(str).apply(lambda x: " ".join(preprocessor.get_ast_sequence(x)))
    texts = df['ast_query'].tolist()
    X = embedder.get_embeddings(texts, batch_size=64)

    kb = build_role_kb_leave_one_out(X, y, labels, num_roles)
    sim_dict = {}
    for r in range(num_roles):
        vecs = kb[r]
        if len(vecs) == 0:
            sim_dict[r] = np.array([1.0])
        else:
            sim_dict[r] = leave_one_out_similarities(vecs)
        print(f"Role {r}: 样本数={len(vecs)}, 相似度均值={sim_dict[r].mean():.4f}, 中位数={np.median(sim_dict[r]):.4f}")

    plot_leave_one_out_distribution(sim_dict, output_dir)

    # 新增：交叉角色相似度分布
    cross_sim = cross_role_similarity(kb, kb, num_roles)
    plot_cross_role_distribution(cross_sim, output_dir)

if __name__ == "__main__":
    main()
