from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.util import batch_to_device
from typing import Literal
import torch.nn as nn
import torch.nn.functional as F
import torch

from models.encoder.multi_negative_loss import MultiNegativeLoss

 
class PovaryoshkaEncoder(nn.Module):
    def __init__(
        self,
        encoder_name="deepvk/USER2-small",
        dtype="float32",
        temperature=0.05,
        matryoshka_dims=[384, 256, 128]
    ):
        super().__init__()
        self.__encoder = SentenceTransformer(encoder_name, model_kwargs={"dtype": dtype})
        base_encoder_loss = MultiNegativeLoss(self.__encoder, temperature)
        self.__encoder_loss = losses.MatryoshkaLoss(
            self.__encoder,
            base_encoder_loss,
            matryoshka_dims
        )

    def forward(
        self,
        query_list: list[str],
        positive_chunk_list: list[str]
    ) -> torch.Tensor:
        query_features = batch_to_device(
            self.__encoder.tokenize(query_list),
            self.__encoder.device
        )
        positive_features = batch_to_device(
            self.__encoder.tokenize(positive_chunk_list),
            self.__encoder.device
        )
        return self.__encoder_loss(
            sentence_features=[query_features, positive_features],
            labels=None
        )
    
    def encode(
        self,
        text_list: list[str],
        prompt_name: Literal["search_query", "search_document"]="search_document",
        is_normalize=True
    ) -> torch.Tensor:
        self.eval()
        with torch.inference_mode():
            embedding_text_list = self.__encoder.encode(
                text_list,
                prompt_name=prompt_name,
                convert_to_tensor=True
            )
            if is_normalize:
                embedding_text_list = F.normalize(embedding_text_list, dim=1)
        return embedding_text_list
