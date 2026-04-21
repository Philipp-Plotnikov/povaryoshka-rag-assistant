import torch
import torch.nn as nn
import numpy as np
import faiss

from models.encoder.encoder import PovaryoshkaEncoder
from models.encoder.encoder_teacher_pool import PovaryoshkaEncoderTeacherPool


class PovaryoshkaEncoderTrainer(nn.Module):
    def __init__(
        self,
        encoder: PovaryoshkaEncoder,
        encoder_teacher_pool: PovaryoshkaEncoderTeacherPool,
        document_list: list[str]
    ):
        super().__init__()
        self.__encoder_teacher_pool = encoder_teacher_pool
        self.__document_list = document_list
        self.__encoder = encoder
        self.__faiss_index: faiss.Index | None = None
    
    def get_encoder(self) -> PovaryoshkaEncoder:
        return self.__encoder
    
    def get_encoder_teacher_pool(self) -> PovaryoshkaEncoderTeacherPool:
        return self.__encoder_teacher_pool

    def forward(
        self,
        batch_query_data: list[tuple[str, torch.Tensor]] # [(str, index_tensor[current_teacher_amount, top_k])]
    ) -> torch.Tensor:
        query_list, positive_chunk_index_tensor = self.__encoder_teacher_pool(batch_query_data)
        positive_chunk_list = []
        for positive_chunk_index in positive_chunk_index_tensor:
            positive_chunk_list.append(
                self.__document_list[positive_chunk_index.item()]
            )
        return self.__encoder(
            query_list=query_list,
            positive_chunk_list=positive_chunk_list
        )
    
    def compute_recall_at_k(
        self,
        query_list: list[str],
        true_chunk_index_list: list[int],
        k=5
    ) -> float:
        self.eval()
        correct_amount = 0
        with torch.inference_mode():
            self.__build_index()
            retrieved_indices_list = self.__search(query_list, top_k=k)
            for i in range(len(query_list)):
                if true_chunk_index_list[i] in retrieved_indices_list[i]:
                    correct_amount += 1
        return correct_amount / len(query_list)
    
    def __build_index(self):
        if self.__faiss_index is not None:
            self.__faiss_index.reset()
        self.eval()
        with torch.inference_mode():
            embedding_list = self.__encoder.encode(self.__document_list).cpu().numpy().astype('float32') 
        embedding_list = np.ascontiguousarray(embedding_list)
        embedding_dimension = embedding_list.shape[1]
        faiss_index = faiss.IndexFlatIP(embedding_dimension)
        faiss_index.add(embedding_list) # type: ignore
        self.__faiss_index = faiss_index

    def __search(
        self,
        query_list: list[str],
        top_k=15
    ) -> list[list[int]]:
        assert self.__faiss_index is not None, "Firstly call __build_index()"
        self.eval()
        with torch.inference_mode():
            query_embedding_list = self.__encoder.encode(query_list, "search_query").cpu().numpy().astype('float32')
        _, indices_list = self.__faiss_index.search(query_embedding_list, top_k) # type: ignore
        return indices_list
