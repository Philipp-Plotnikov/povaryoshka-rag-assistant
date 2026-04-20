import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiNegativeLoss(nn.Module):
    def __init__(self, encoder: nn.Module, temperature=0.05):
        super().__init__()
        self.__encoder = encoder
        self.__temperature = temperature

    def forward(self, sentence_features, labels=None) -> torch.Tensor:
        query_features, positive_features = sentence_features
        query_embeddings = self.__encoder(query_features)["sentence_embedding"] # [B, D]
        pos_embeddings   = self.__encoder(positive_features)["sentence_embedding"]
        query_embeddings = F.normalize(query_embeddings, dim=1)
        pos_embeddings   = F.normalize(pos_embeddings, dim=1)
        scores = torch.matmul(query_embeddings, pos_embeddings.T) / self.__temperature
        labels = torch.arange(scores.size(0), device=scores.device)
        return F.cross_entropy(scores, labels)
