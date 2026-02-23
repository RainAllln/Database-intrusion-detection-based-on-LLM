import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, classification_report

def plot_confusion_matrix(y_true, y_pred, output_dir, labels=None, filename='confusion_matrix.png', cmap='Blues', title='Confusion Matrix'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path)
    plt.close()
    print(f"图表已保存: {save_path}")

def plot_roc_curve(y_true, scores, output_dir, filename='roc_curve.png'):
    fpr, tpr, _ = roc_curve(y_true, -scores)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc_score(y_true, -scores):.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path)
    plt.close()
    print(f"图表已保存: {save_path}")

def plot_score_distribution(y_true, scores, output_dir, filename='score_dist.png', title='Anomaly Score Distribution'):
    plt.figure(figsize=(8, 5))
    df_res = pd.DataFrame({'Score': scores, 'Label': y_true})
    sns.histplot(data=df_res, x='Score', hue='Label', element="step", stat="density", common_norm=False, bins=30)
    plt.title(title)
    plt.xlabel('Anomaly Score (Lower is more anomalous)')
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path)
    plt.close()
    print(f"图表已保存: {save_path}")

def plot_loss_curve(loss_history, output_dir, filename='training_loss.png', title='Training Loss'):
    plt.figure()
    plt.plot(loss_history, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path)
    plt.close()
    print(f"图表已保存: {save_path}")

def plot_rag_similarity_distribution(rag_sim_distributions, output_dir, filename='rag_sim_distribution.png'):
    """
    rag_sim_distributions: dict, key=role, value=list of similarity scores
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(10, 6))
    for r, sims in rag_sim_distributions.items():
        sns.histplot(sims, bins=30, kde=True, label=f'Role {r}', stat='density', alpha=0.5)
    plt.xlabel('Max Similarity to Role Knowledge Base')
    plt.ylabel('Density')
    plt.title('RAG Similarity Distribution for Each Role (Train, Label=0)')
    plt.legend()
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path)
    plt.close()
    print(f"RAG相似度分布图已保存: {save_path}")

def to_camel_case(snake_str):
    """将下划线命名法转换为驼峰命名法."""
    components = snake_str.split('_')
    return components[0] + ''.join(x.title() for x in components[1:])

def write_experiment_report(report_path, title, model_paras, note, content):
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"=== {title} ===\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("\n--- Model Parameters ---\n")
        f.write(model_paras + "\n")
        f.write("\n--- Experiment Details ---\n")
        f.write(note + "\n")
        f.write("\n--- Experiment Results ---\n")
        f.write(content)
    print(f"实验参数与结果已写入: {report_path}")

def write_detail_log(detail_path, df, preds, probs, rag_sims, pred_layer2, actual_label, sample_limit=200):
    with open(detail_path, 'w', encoding='utf-8') as f:
        headers = ["ID", "True_Role", "Pred_Role", "Result", "Confidence", "RAG_Sim", "SQL_Snippet", "pred", "actual"]
        f.write(
            f"{headers[0]:<5} | {headers[1]:<10} | {headers[2]:<10} | {headers[3]:<8} | {headers[4]:<10} | {headers[5]:<8} | {headers[6]:<55} | {headers[7]:<5} | {headers[8]:<6}\n")
        f.write("-" * 160 + "\n")
        for i in range(sample_limit):
            query_clip = df.loc[i, 'query'][:50].replace('\n', ' ') + "..."
            actual_role = df.loc[i, 'role']
            pred_role = preds[i].item()
            confidence = probs[i][pred_role]
            is_match = (actual_role == pred_role)
            res_mark = "MATCH" if is_match else "MISMATCH"
            rag_sim = rag_sims[i][actual_role]
            pred_val = pred_layer2[i]
            actual_val = actual_label[i]
            f.write(
                f"{i:<5} | {actual_role:<10} | {pred_role:<10} | {res_mark:<8} | {confidence:.4f}     | {rag_sim:.4f}   | {query_clip:<55} | {pred_val:<5} | {actual_val:<6}\n")
    print(f"详细推理日志已写入: {detail_path}")

