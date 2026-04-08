from sentence_transformers import SentenceTransformer, losses
import faiss
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Literal
from teacher_retriever_pool import TeacherRetrieverPool


class MultiNegativeLoss(nn.Module):
    def __init__(self, encoder, temperature=0.05):
        super().__init__()
        self.encoder = encoder
        self.temperature = temperature


    def forward(self, sentence_features, labels=None):
        query_features, positive_features = sentence_features
        query_embeddings = self.encoder(query_features)["sentence_embedding"] # [B, D]
        pos_embeddings   = self.encoder(positive_features)["sentence_embedding"]
        query_embeddings = F.normalize(query_embeddings, dim=1)
        pos_embeddings   = F.normalize(pos_embeddings, dim=1)
        scores = torch.matmul(query_embeddings, pos_embeddings.T) / self.temperature
        labels = torch.arange(scores.size(0), device=scores.device)
        return F.cross_entropy(scores, labels)

class PovaryoshkaRetriever(nn.Module):
  def __init__(
    self,
    document_list: list[str],
    teacher_retriever_pool: TeacherRetrieverPool,
    device: Literal['cpu', 'cuda', 'mps'] = 'cuda',
    encoder_name="deepvk/USER2-small",
    dtype="float32",
    temperature=0.05,
    matryoshka_dims=[384, 256, 128],
    nlist=100
  ):
    super().__init__()
    self.temperature = temperature
    self.document_list = document_list
    self.retriever_index = None
    self.DTYPE = dtype
    self.encoder = SentenceTransformer(encoder_name, device=device, model_kwargs={"dtype": dtype})
    base_encoder_loss = MultiNegativeLoss(self.encoder, temperature)
    self.encoder_loss = losses.MatryoshkaLoss(
      self.encoder,
      base_encoder_loss,
      matryoshka_dims
    )
    self.teacher_retriever_pool = teacher_retriever_pool
    self.NLIST = nlist


  def forward(
    self,
    batch_query_data: list[tuple[str, torch.Tensor, torch.Tensor]] # [(str, index_tensor[current_teacher_amount, top_k], ranking_tensor[current_teacher_amount, top_k])]
  ) -> torch.Tensor:
    assert self.document_list is not None, "Firstly, initialize document_list field"
    encoder_device = next(self.encoder.parameters()).device
    query_list, positive_chunk_index_tensor = self.teacher_retriever_pool(batch_query_data)
    positive_chunk_list = []

    for positive_chunk_index in positive_chunk_index_tensor:
      positive_chunk_list.append(
        self.document_list[positive_chunk_index.item()]
      )

    query_features = self.to_device(
      self.encoder.tokenize(query_list), encoder_device
    )
    positive_features = self.to_device(
      self.encoder.tokenize(positive_chunk_list), encoder_device
    )
    return self.encoder_loss(
      sentence_features=[query_features, positive_features],
      labels=None
    )


  def to_device(self, features, device):
    return {k: v.to(device) for k, v in features.items()}


  def get_encoded_query_tensor(
    self,
    query_list: list[str],
    is_normalize=True
  ) -> torch.Tensor:
    self.eval()
    with torch.inference_mode():
      return self._get_encoded_text_tensor(query_list, True, is_normalize)


  def get_encoded_document_tensor(
    self,
    document_list: list[str],
    is_normalize=True
  ) -> torch.Tensor:
    self.eval()
    with torch.inference_mode():
      return self._get_encoded_text_tensor(document_list, is_normalize)
      

  def _get_encoded_text_tensor(
    self,
    text_list: list[str],
    is_query=False,
    is_normalize=True
  ) -> torch.Tensor:
    embedding_text_list = self.encoder.encode(
      text_list,
      prompt_name="search_query" if is_query else "search_document",
      convert_to_tensor=True,
      truncate_dim=384
    )
    if is_normalize:
      embedding_text_list = F.normalize(embedding_text_list, dim=1)
    return embedding_text_list


  def build_index(self):
    self.eval()
    with torch.inference_mode():
      # TODO: Check about type
      embedding_list = self._get_encoded_text_tensor(self.document_list).cpu().numpy().astype('float32') 
    embedding_list = np.ascontiguousarray(embedding_list)
    embedding_dimension = embedding_list.shape[1]
    retriever_index = faiss.IndexFlatIP(embedding_dimension)
    retriever_index.add(embedding_list) # type: ignore
    self.retriever_index = retriever_index


  def search(
    self,
    query_list: list[str],
    top_k=15
  ) -> list[list[int]]:
    assert self.retriever_index is not None, "Firstly call build_index()"
    self.eval()
    with torch.inference_mode():
      query_embedding_list = self._get_encoded_text_tensor(
        query_list,
        True
      ).cpu().numpy().astype('float32')
    _, indices_list = self.retriever_index.search(query_embedding_list, top_k) # type: ignore
    return indices_list


  def save(self, path: str):
    self.encoder.save(path)


  def load(self, path: str):
    self.encoder = SentenceTransformer(path)
