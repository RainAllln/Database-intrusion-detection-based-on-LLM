import torch

def Layer2Classifier(model_name="mlp", input_dim=772, num_roles=4):
    if model_name == "mlp":
        from models.mlp_classifier import MLPClassifier
        return MLPClassifier(input_dim=input_dim, num_roles=num_roles)
    else:
        raise ValueError(f"未知模型: {model_name}")
