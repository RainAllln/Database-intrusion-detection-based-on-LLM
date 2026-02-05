import torch.nn as nn
import torch.nn.functional as F

class MLPClassifier(nn.Module):
    def __init__(self, input_dim=772, num_roles=4):
        super(MLPClassifier, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_roles)
        )

    def forward(self, sql_embedding):
        logits = self.main(sql_embedding)
        return F.softmax(logits, dim=1)
