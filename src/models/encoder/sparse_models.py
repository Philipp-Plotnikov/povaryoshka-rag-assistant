import torch
import numpy as np
from rank_bm25 import BM25Okapi
import re


class SparseModel:
    def __init__(self, chunk_list: list[dict]):
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
