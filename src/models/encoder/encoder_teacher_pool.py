import torch
import torch.nn as nn


class PovaryoshkaEncoderTeacherPool(nn.Module):
    def __init__(self, teacher_amount=2, top_k=12, rrf_K=60):
        super().__init__()
        self.RRF_K = rrf_K
        self.top_k = top_k
        self.total_teacher_amount = teacher_amount
        self.teacher_model_dict = {}
        self.weight_tensor = nn.Parameter(torch.ones(teacher_amount))


    def forward(
        self,
        query_data_batch: list[tuple[str, torch.Tensor]] # [(str, index_tensor[current_teacher_amount, top_k])]
    ) -> tuple[list[str], torch.Tensor]: # (list of query, positive_document_index_tensor)
        device = next(self.parameters()).device
        batch_size = len(query_data_batch)
        query_list = []    
        fuse_ranked_index_tensor_list = []

        for query_data in query_data_batch:
            query_list.append(query_data[0])
            ranked_index_tensor = self.get_fuse_ranked_index_tensor(query_data[1])  # [top_k]
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
    def get_fuse_ranked_index_tensor(self, index_tensor: torch.Tensor) -> torch.Tensor:
        ranking_tensor = torch.arange(1, index_tensor.size(1) + 1, device=index_tensor.device)
        ranking_tensor = ranking_tensor.unsqueeze(0).expand(index_tensor.size(0), -1)
        teacher_weight_tensor = self.weight_tensor[:index_tensor.size(0)].view(-1, 1)  # [teacher_amount, 1]
        rrf_score_tensor = teacher_weight_tensor / (self.RRF_K + ranking_tensor)  # [teacher_amount, top_k]
        doc_ids = index_tensor.reshape(-1)                     # [teacher_amount*top_k]
        rrf_score_tensor = rrf_score_tensor.reshape(-1)          # [teacher_amount*top_k]
        unique_doc_ids, inverse_indices = torch.unique(doc_ids, return_inverse=True)
        fused_score_tensor = torch.zeros_like(unique_doc_ids, dtype=rrf_score_tensor.dtype)
        fused_score_tensor.scatter_add_(0, inverse_indices, rrf_score_tensor)
        sorted_idx = torch.argsort(fused_score_tensor, descending=True)
        return unique_doc_ids[sorted_idx][:index_tensor.size(1)]
    

    def __len__(self) -> int:
        return len(self.teacher_model_dict)
