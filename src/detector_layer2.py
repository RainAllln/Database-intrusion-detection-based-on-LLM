import torch
import torch.nn as nn
import torch.nn.functional as F

class Layer2Classifier(nn.Module):
    def __init__(self, input_dim=768, num_roles=4):
        super(Layer2Classifier, self).__init__()
        # 升级为 3 层 MLP，带 BatchNorm 和 Dropout
        self.main = nn.Sequential(
            # Layer 1
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),       # 标准化，让特征分布更稳
            nn.ReLU(),
            nn.Dropout(0.3),           # 防止死记硬背

            # Layer 2
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Output Layer to Role Scores
            nn.Linear(128, num_roles)
        )

    def forward(self, sql_embedding):
        logits = self.main(sql_embedding)
        return F.softmax(logits, dim=1)