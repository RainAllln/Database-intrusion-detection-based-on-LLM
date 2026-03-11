import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from src.preprocess import SQLPreprocessor
from src.feature import SQLEmbedder
from src.rag import build_role_knowledge_base_faiss_l2, get_top1_l2_distances_faiss


def _predict_roles_by_kb_top1_l2(X_emb_np: np.ndarray, kb, num_roles: int) -> np.ndarray:
    """
    对每条 SQL embedding，计算其到每个角色 KB 的 top1 L2 距离；
    取距离最小的角色作为预测。
    """
    preds = np.empty((X_emb_np.shape[0],), dtype=np.int64)
    for i in range(X_emb_np.shape[0]):
        dists = get_top1_l2_distances_faiss(X_emb_np[i], kb, num_roles=num_roles)  # (num_roles,)
        preds[i] = int(np.argmin(dists))
    return preds


def _per_role_accuracy(y_true: np.ndarray, y_pred: np.ndarray, num_roles: int):
    rows = []
    for r in range(num_roles):
        mask = (y_true == r)
        n = int(mask.sum())
        acc_r = float((y_pred[mask] == y_true[mask]).mean()) if n > 0 else float("nan")
        rows.append({"role": r, "n": n, "acc": acc_r})
    return pd.DataFrame(rows)


def test_rag(
    extractor_type: str = "distilbert",
    batch_size_embed: int = 128,
    test_size: float = 0.2,
    random_state: int = 42,
    num_roles: int = 8,
):
    # 1) 加载数据
    data_path = os.path.join(project_root, 'data', 'custom', 'complex_dataset_v2.csv')
    print(f"正在读取数据集: {data_path}")
    try:
        df = pd.read_csv(data_path, encoding='utf-8')
    except (UnicodeError, FileNotFoundError, Exception):
        try:
            df = pd.read_csv(data_path, encoding='gbk')
        except Exception:
            df = pd.read_csv(data_path, encoding='latin-1')

    df = df.dropna(subset=['query', 'role', 'Label']).reset_index(drop=True)
    print(f"有效样本数: {len(df)}")

    # 2) 预处理
    preprocessor = SQLPreprocessor()
    df['clean_query'] = df['query'].astype(str).apply(preprocessor.normalize)

    y_roles = df['role'].values.astype(int)
    labels_global = df['Label'].values.astype(int)

    # 3) embedding（全量先算出来，后面再切分；也可先切分再分别算）
    embedder = SQLEmbedder(extractor_type=extractor_type)
    print(f"正在提取 embedding: extractor_type={extractor_type} ...")
    X_emb = embedder.get_embeddings(df['clean_query'].values, batch_size=batch_size_embed)
    if X_emb is None:
        raise RuntimeError("get_embeddings 返回 None，请检查实现。")

    X_emb = np.asarray(X_emb, dtype=np.float32)

    # 4) 切分（避免 KB 泄漏：KB 只用训练集构建）
    X_train, X_test, y_train, y_test, lab_train, lab_test = train_test_split(
        X_emb,
        y_roles,
        labels_global,
        test_size=test_size,
        random_state=random_state,
        stratify=y_roles,
    )

    # 5) 构建角色知识库（仅 Label==0）
    print("正在基于训练集(Label==0)构建 FAISS 角色知识库(IndexFlatL2)...")
    kb = build_role_knowledge_base_faiss_l2(
        X_train=torch.tensor(X_train, dtype=torch.float32),
        y_train=torch.tensor(y_train, dtype=torch.long),
        labels_train=torch.tensor(lab_train, dtype=torch.long),
        num_roles=num_roles,
    )

    # 6) 检索预测：选择 top1 L2 距离最小的角色
    print("正在对测试集做 KB 检索预测(选择最小 L2 距离的角色)...")
    y_pred = _predict_roles_by_kb_top1_l2(X_test, kb, num_roles=num_roles)

    # 7) 评估：整体准确率 + 分角色准确率
    acc = accuracy_score(y_test, y_pred)
    print("\n=== RAG/向量库 top1(L2) 角色预测评估 ===")
    print(f"extractor_type: {extractor_type}")
    print(f"overall acc: {acc:.4f}")

    per_role = _per_role_accuracy(y_test, y_pred, num_roles=num_roles)
    print("\n--- per-role accuracy ---")
    # 只打印关键列，按 role 排序
    print(per_role.sort_values("role")[["role", "n", "acc"]].to_string(index=False))

    print("\n--- classification report ---")
    print(classification_report(y_test, y_pred, target_names=[f"R{i}" for i in range(num_roles)]))

    return acc, per_role, (y_test, y_pred)


if __name__ == "__main__":
    test_rag()
