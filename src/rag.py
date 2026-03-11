import numpy as np
import torch
import faiss

def l2_normalize(vecs: np.ndarray) -> np.ndarray:
    """
    对向量做 L2 归一化。
    常用于：将嵌入向量归一化后再进行相似度 / 距离计算。
    """
    norm = np.linalg.norm(vecs, axis=1, keepdims=True)
    norm[norm == 0] = 1
    return vecs / norm

def build_role_knowledge_base_faiss_l2(X_train, y_train, labels_train, num_roles):
    """
    使用 FAISS L2 距离，为每个角色构建一个知识库索引。
    仅使用 Label == 0 的正常样本。
    X_train: (N, D) torch.Tensor 或 np.ndarray
    y_train: (N,) 对应角色 ID
    labels_train: (N,) 全局 Label（0 正常 / 2 越权 等）
    """
    mask = (labels_train == 0)
    kb = {}

    X_train_np = X_train.cpu().numpy() if isinstance(X_train, torch.Tensor) else X_train
    y_train_np = y_train.cpu().numpy() if isinstance(y_train, torch.Tensor) else y_train
    dim = X_train_np.shape[1]

    for r in range(num_roles):
        idx = (y_train_np[mask] == r)
        vecs = X_train_np[mask][idx]
        index = faiss.IndexFlatL2(dim)
        if vecs.shape[0] == 0:
            # 若某个角色在正常样本中没有数据，放一个全零向量占位，避免 FAISS 报错
            index.add(np.zeros((1, dim), dtype="float32"))
        else:
            index.add(vecs.astype("float32"))
        kb[r] = index

    return kb


def get_top1_l2_distances_faiss(sql_emb, kb, num_roles):
    """
    给定一条 SQL 的向量 embedding（1D 或 (D,)），
    对每个角色知识库计算 Top-1 L2 距离，返回 shape=(num_roles,) 的数组。
    """
    emb = sql_emb.astype("float32").reshape(1, -1)
    dists = []
    for r in range(num_roles):
        D, _ = kb[r].search(emb, 1)
        dist = float(D[0][0])
        dists.append(dist)
    return np.array(dists)


def get_enhanced_rag_features(sql_emb, kb, num_roles, k=5):
    """
    2.1 用“前 k 个邻居”的统计特征 + 2.2 归一化相似度分布
    返回 (num_roles * 3 + num_roles) 维度的特征向量
    """
    emb = sql_emb.astype("float32").reshape(1, -1)
    stats_features = []
    min_distances = []

    for r in range(num_roles):
        # 搜索前 k 个最近邻
        D, _ = kb[r].search(emb, k)
        dists = D[0]
        
        # 统计特征：最小值, 均值, 方差
        d_min = float(dists[0])
        d_mean = float(np.mean(dists))
        d_var = float(np.std(dists)) # 使用标准差
        
        stats_features.extend([d_min, d_mean, d_var])
        min_distances.append(d_min)

    # 2.2 转换为归一化后的相似度分布 (Softmax style)
    min_distances = np.array(min_distances)
    
    # 调低温度参数 T=0.05使预测分布更显著（配合微调后的 Embedding）
    T = 0.05 
    norm_dist = (min_distances - np.min(min_distances)) / (np.mean(min_distances) + 1e-6)
    exp_dist = np.exp(-norm_dist / T)
    prob_dist = exp_dist / (np.sum(exp_dist) + 1e-10)

    return np.concatenate([stats_features, prob_dist])

