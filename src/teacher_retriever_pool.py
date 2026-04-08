import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
import re
import numpy as np
from rank_bm25 import BM25Okapi
import faiss
from typing import Literal


batch_size = 60

class BM25Model:
    def __init__(self, chunk_list: list):
        text_chunk_list = [chunk['text'] for chunk in chunk_list]
        token_list = []
        for chunk in text_chunk_list:
            word_list = re.findall(r'\w+', chunk.lower())
            token_list.append(word_list)
        self.model = BM25Okapi(token_list)


    def search(self, question: str, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        tokenized_query = re.findall(r'\w+', question.lower())
        score_list = self.model.get_scores(tokenized_query)
        # TODO: dont like the copy
        index_list = torch.tensor(np.argsort(score_list)[-k:][::-1].copy())
        return index_list, torch.tensor(score_list)[index_list]

class DenseModel:
    def __init__(self, chunk_list: list, model_name="deepvk/USER2-small", device: Literal['cpu', 'cuda', 'mps']='cuda'):
        self.model = SentenceTransformer(model_name, device=device)
        text_chunk_list = [chunk['text'] for chunk in chunk_list]
        self.embedding_list = self.model.encode(
            text_chunk_list,
            prompt_name="search_document",
            show_progress_bar=True,
            batch_size=batch_size
        )
        faiss.normalize_L2(self.embedding_list)
        self.index = faiss.IndexFlatIP(self.embedding_list.shape[1])
        self.index.add(self.embedding_list.astype('float32')) # type: ignore


    def search(self, question: str, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        query_embedding = self.model.encode([question], prompt_name="search_query")
        faiss.normalize_L2(query_embedding)
        scores, indices = self.index.search(query_embedding.astype('float32'), k) # type: ignore
        return torch.tensor(indices[0]), torch.tensor(scores[0])

class TeacherRetrieverPool(nn.Module):
    def __init__(self, teacher_amount=2, top_k=12, rrf_K=60):
        super().__init__()
        self.RRF_K = rrf_K
        self.top_k = top_k
        self.total_teacher_amount = teacher_amount
        self.teacher_model_dict = {}
        self.weight_tensor = nn.Parameter(torch.ones(teacher_amount))


    def forward(
        self,
        query_data_batch: list[tuple[str, torch.Tensor, torch.Tensor]] # [(str, index_tensor[current_teacher_amount, top_k], ranking_tensor[current_teacher_amount, top_k])]
    ) -> tuple[list[str], torch.Tensor]: # (list of query, positive_document_index_tensor)
        device = next(self.parameters()).device
        batch_size = len(query_data_batch)
        query_list = []    
        fuse_ranked_index_tensor_list = []

        for query_data in query_data_batch:
            query_list.append(query_data[0])
            ranked_index_tensor = self.get_fuse_ranked_index_tensor(query_data[1], query_data[2])
            fuse_ranked_index_tensor_list.append(ranked_index_tensor)
        fused_ranked_index_batch = torch.stack(fuse_ranked_index_tensor_list)  # [batch, top_k]
        
        positive_document_position_tensor = torch.randint(0, 5, (batch_size,))
        positive_document_index_tensor = fused_ranked_index_batch[torch.arange(batch_size), positive_document_position_tensor]
        return query_list, positive_document_index_tensor.to(device)


    def add_teacher(self, teacher_name: str, teacher_model):
        if len(self) < self.total_teacher_amount:
            self.teacher_model_dict[teacher_name] = teacher_model


    def get_index_and_ranking_tensor(self, question_list: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        index_tensor = torch.zeros((len(question_list), self.total_teacher_amount, self.top_k), dtype=torch.long)
        ranking_tensor = torch.zeros((len(question_list), self.total_teacher_amount, self.top_k))
        for question_index, question in enumerate(question_list):
            for teacher_index, (_, teacher_model) in enumerate(self.teacher_model_dict.items()):
                teacher_index_tensor, teacher_score_tensor = teacher_model.search(question, self.top_k)
                index_tensor[question_index, teacher_index] = teacher_index_tensor
                ranking_tensor[question_index, teacher_index] = teacher_score_tensor
        return index_tensor, ranking_tensor


    # ranking_tensor shape is [current_teacher_amount, top_k]
    def get_fuse_ranked_index_tensor(self, index_tensor: torch.Tensor, ranking_tensor: torch.Tensor) -> torch.Tensor:
        teacher_weight_tensor = self.weight_tensor[:index_tensor.size(0)].view(-1, 1)  # [teacher_amount, 1]
        rrf_score_tensor = teacher_weight_tensor / (self.RRF_K + ranking_tensor)
        flat_index_tensor = index_tensor.flatten()       # [teacher_amount*top_k]
        flat_score_tensor = rrf_score_tensor.flatten()         # [teacher_amount*top_k]
        fused_score_tensor = torch.zeros(int(flat_index_tensor.max().item() + 1), dtype=torch.float)
        fused_score_tensor = fused_score_tensor.scatter_add(0, flat_index_tensor, flat_score_tensor)
        sorted_document_index_tensor = torch.argsort(fused_score_tensor, descending=True)
        return sorted_document_index_tensor
    

    def __len__(self) -> int:
        return len(self.teacher_model_dict)