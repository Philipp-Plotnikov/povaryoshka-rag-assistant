import torch
import faiss
from sentence_transformers import SentenceTransformer

# TODO: Define device dynamically
# TODO: Think about the best batch size value
class DenseModel:
    def __init__(
        self,
        chunk_list: list[dict],
        batch_size=60,
        model_name="deepvk/USER2-small",
        device='cuda'
    ):
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
