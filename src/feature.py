import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel


class SQLEmbedder:
    def __init__(self, model_name="distilbert-base-uncased"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        
        # 打印明确的设备信息
        if self.device.type == 'cuda':
            print(f"模型利用 GPU 加速中: {torch.cuda.get_device_name(0)}")
        else:
            print("未检测到 GPU，正在使用 CPU")

    def get_embeddings(self, texts, batch_size=16):
        """捕捉SQL的深层语义特征"""
        all_embeddings = []
        
        # 确保输入可迭代
        if isinstance(texts, str):
            texts = [texts]
            
        for i in range(0, len(texts), batch_size):
            batch_texts = list(texts[i:i + batch_size])
            inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # 取 [CLS] token (位置0) 作为句向量
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                all_embeddings.append(batch_embeddings)
        
        if not all_embeddings:
            return np.array([])
            
        return np.vstack(all_embeddings)

