import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

class CodeBERTFeatureExtractor:
    """基于 CodeBERT (microsoft/codebert-base) 的特征提取"""
    def __init__(self, model_name: str = "microsoft/codebert-base"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

        if self.device.type == "cuda":
            print(f"模型利用 GPU 加速中: {torch.cuda.get_device_name(0)}")
        else:
            print("未检测到 GPU，正在使用 CPU")

    def get_embeddings(self, texts, batch_size: int = 16):
        """捕捉SQL的深层语义特征（基于代码预训练模型 CodeBERT）"""
        all_embeddings = []
        if isinstance(texts, str):
            texts = [texts]

        for i in range(0, len(texts), batch_size):
            batch_texts = list(texts[i : i + batch_size])
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                
                # 改为 Mean Pooling: 对所有 token 的向量按 attention_mask 做平均
                hidden_states = outputs.last_hidden_state
                attention_mask = inputs['attention_mask']
                
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                
                batch_embeddings = (sum_embeddings / sum_mask).cpu().numpy()
                
                all_embeddings.append(batch_embeddings)

        if not all_embeddings:
            return np.array([])
        return np.vstack(all_embeddings)

