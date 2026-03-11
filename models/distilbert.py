import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
import os

class SQLDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)

class DistilBERTFeatureExtractor:
    """基于 DistilBERT 模型的特征提取与微调"""
    def __init__(self, model_name="distilbert-base-uncased", local_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        load_path = local_path if local_path and os.path.exists(local_path) else model_name
        self.tokenizer = AutoTokenizer.from_pretrained(load_path)
        self.model = AutoModel.from_pretrained(load_path).to(self.device)
        
        if self.device.type == 'cuda':
            print(f"模型利用 GPU 加速中: {torch.cuda.get_device_name(0)}")
        else:
            print("未检测到 GPU，正在使用 CPU")

    def fine_tune(self, texts, labels, num_classes, output_dir):
        # 当前无微调需求，提供占位实现，防止误调用时报错
        print("DistilBERTFeatureExtractor.fine_tune 被调用，但当前测试无微调需求，跳过训练。")
        return

    def get_embeddings(self, texts, batch_size=16):
        """利用 DistilBERT 提取句向量（平均池化 last_hidden_state）"""
        self.model.eval()
        all_embeddings = []

        # 统一成 list[str]
        if isinstance(texts, (list, tuple, np.ndarray)):
            text_list = [str(t) for t in texts]
        else:
            text_list = [str(texts)]

        with torch.no_grad():
            for i in range(0, len(text_list), batch_size):
                batch_texts = text_list[i:i + batch_size]
                encodings = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt",
                ).to(self.device)

                outputs = self.model(**encodings)
                # [batch, seq_len, hidden]
                last_hidden = outputs.last_hidden_state
                # 简单平均池化为句向量
                emb = last_hidden.mean(dim=1)  # [batch, hidden]
                all_embeddings.append(emb.cpu().numpy())

        if not all_embeddings:
            hidden_size = getattr(self.model.config, "hidden_size", 768)
            return np.zeros((0, hidden_size), dtype=np.float32)

        return np.vstack(all_embeddings)

class DistilBERTClassifierDataset(Dataset):
    """用于 DistilBERT 分类器的 Dataset，输入为文本 + RAG 数值特征（编码为附加文本）"""
    def __init__(self, tokenizer, texts, rag_features, labels=None, max_length=128):
        self.tokenizer = tokenizer
        self.texts = texts
        self.rag_features = rag_features
        self.labels = labels
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        sql_text = str(self.texts[idx])
        rag_vec = self.rag_features[idx]
        # 将 RAG 特征简单序列化为文本，拼在后面
        rag_str = " ".join([f"{v:.4f}" for v in rag_vec])
        combined = f"{sql_text} [RAG] {rag_str}"
        enc = self.tokenizer(
            combined,
            truncation=True,
            padding="max_length",
            max_length=self.max_length
        )
        item = {k: torch.tensor(v) for k, v in enc.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(int(self.labels[idx]))
        return item

class DistilBERTClassifierModel:
    """使用 DistilBERT 自带分类头的角色分类器"""
    def __init__(self, num_roles: int, device=None, model_name: str = "distilbert-base-uncased", local_path: str = None):
        self.num_roles = num_roles
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        load_path = local_path if local_path and os.path.exists(local_path) else model_name
        self.tokenizer = AutoTokenizer.from_pretrained(load_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            load_path,
            num_labels=num_roles
        ).to(self.device)
        self.trainer = None
        if self.device.type == 'cuda':
            print(f"DistilBERTClassifierModel 使用 GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("DistilBERTClassifierModel 使用 CPU")

    def train_model(self, texts, rag_features, y_train, training_hparams: dict, output_dir: str = None):
        output_dir = output_dir or "./distilbert_clf_output"
        os.makedirs(output_dir, exist_ok=True)

        train_dataset = DistilBERTClassifierDataset(
            tokenizer=self.tokenizer,
            texts=texts,
            rag_features=rag_features,
            labels=y_train,
            max_length=training_hparams.get("max_length", 128),
        )

        # 只使用兼容的 TrainingArguments 参数，避免 evaluation_strategy 等老版本不存在的字段
        args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=training_hparams.get("num_train_epochs", 3),
            per_device_train_batch_size=training_hparams.get("per_device_train_batch_size", 16),
            per_device_eval_batch_size=training_hparams.get("per_device_eval_batch_size", 32),
            learning_rate=training_hparams.get("learning_rate", 5e-5),
            weight_decay=training_hparams.get("weight_decay", 0.01),
            warmup_ratio=training_hparams.get("warmup_ratio", 0.1),
            logging_steps=training_hparams.get("logging_steps", 50),
        )

        self.trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_dataset,
        )

        train_result = self.trainer.train()
        # loss_history 简单返回每步 loss（如果可用）
        try:
            metrics = train_result.metrics
            return [float(metrics.get("train_loss", 0.0))]
        except Exception:
            return None

    def predict_proba(self, texts, rag_features):
        if self.trainer is None:
            # 构造临时 trainer 以便使用其预测接口
            dummy_args = TrainingArguments(
                output_dir="./tmp_distilbert_clf",
                per_device_eval_batch_size=32,
            )
            self.trainer = Trainer(
                model=self.model,
                args=dummy_args,
            )

        dataset = DistilBERTClassifierDataset(
            tokenizer=self.tokenizer,
            texts=texts,
            rag_features=rag_features,
            labels=None,
        )
        predictions = self.trainer.predict(dataset)
        logits = predictions.predictions
        probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
        y_pred = probs.argmax(axis=1).astype(np.int64)
        return y_pred, probs

    def get_hyperparams_str(self, training_hparams: dict, extra_params: dict = None) -> str:
        extra_params = extra_params or {}
        hp_str = (
            f"DistilBERTClassifierModel("
            f"num_train_epochs={training_hparams.get('num_train_epochs', 3)}, "
            f"lr={training_hparams.get('learning_rate', 5e-5)}, "
            f"batch_size={training_hparams.get('per_device_train_batch_size', 16)})"
        )
        for k, v in extra_params.items():
            hp_str += f", {k}={v}"
        return hp_str

