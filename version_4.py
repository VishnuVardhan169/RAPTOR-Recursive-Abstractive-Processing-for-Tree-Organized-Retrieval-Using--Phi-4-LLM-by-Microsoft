import torch
from llama_cpp import Llama
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Union, Literal
import os

class LlamaRaptorGGUF:
    def __init__(
        self,
        model_path: str = "models/phi-4-Q6_K.gguf",
        n_ctx: int = 2048,
        n_gpu_layers: int = 35,
        index_type: Union[Literal["flat"], Literal["hnsw"], Literal["ivf_hnsw"]] = "hnsw",
        nlist: int = 100,
        hnsw_m: int = 32,
        use_gpu: bool = True
    ):
        print("Initializing Llama GGUF model...")

        self.index_type = index_type
        self.nlist = nlist
        self.hnsw_m = hnsw_m
        self.use_gpu = use_gpu and torch.cuda.is_available()

        # Initialize GGUF model
        try:
            self.llm = Llama(
                model_path=model_path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers
            )
            print("Llama GGUF model loaded successfully!")
        except Exception as e:
            print(f"Error loading Llama GGUF model: {e}")
            raise

        # Initialize FAISS retriever
        print("Setting up FAISS retriever...")
        self.embedding_model = SentenceTransformer('bert-large-nli-mean-tokens')
        if self.use_gpu:
            self.embedding_model = self.embedding_model.to("cuda")
            self.res = faiss.StandardGpuResources()
        else:
            self.res = None
        self.index = None
        self.documents = []
        self.is_initialized = False

    def _create_index(self, dim: int):
        print(f"Creating {self.index_type} index...")
        if self.index_type == "hnsw":
            index = faiss.IndexHNSWFlat(dim, self.hnsw_m)
            index.hnsw.efConstruction = 40
            index.hnsw.efSearch = 16
            return index
        elif self.index_type == "ivf_hnsw":
            quantizer = faiss.IndexHNSWFlat(dim, self.hnsw_m)
            index = faiss.IndexIVFFlat(quantizer, dim, self.nlist)
            if self.use_gpu:
                index = faiss.index_cpu_to_gpu(self.res, 0, index)
            return index
        else:  # "flat"
            index = faiss.IndexFlatL2(dim)
            if self.use_gpu:
                index = faiss.index_cpu_to_gpu(self.res, 0, index)
            return index

    def add_documents(self, documents: List[str], batch_size: int = 32):
        if not documents:
            raise ValueError("Documents list cannot be empty")

        print(f"Processing {len(documents)} documents...")
        self.documents = documents

        # Batch processing for embeddings
        all_embeddings = []
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            with torch.no_grad():
                embeddings = self.embedding_model.encode(
                    batch,
                    convert_to_tensor=True,
                    show_progress_bar=True
                )
                if self.use_gpu:
                    embeddings = embeddings.cpu()
                if torch.is_tensor(embeddings):
                    embeddings = embeddings.numpy()
                all_embeddings.append(embeddings)

        embeddings = np.vstack(all_embeddings)

        # Initialize and train index
        dim = embeddings.shape[1]
        self.index = self._create_index(dim)

        if self.index_type == "ivf_hnsw":
            print("Training IVF index...")
            self.index.train(embeddings)

        # Add vectors to index
        self.index.add(embeddings)
        print("Documents indexed successfully!")
        self.is_initialized = True

        # Save index
        if self.use_gpu and self.index_type != "hnsw":
            faiss.write_index(
                faiss.index_gpu_to_cpu(self.index),
                'faiss_index.bin'
            )
        else:
            faiss.write_index(self.index, 'faiss_index.bin')

    def generate_answer(
        self,
        query: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        k: int = 3,
        nprobe: int = 10
    ) -> str:
        if not self.is_initialized:
            raise ValueError("Index not initialized. Please add documents first using add_documents()")

        # Retrieve relevant documents
        query_embedding = self.embedding_model.encode([query])
        if self.use_gpu:
            query_embedding = query_embedding.cpu()
        if torch.is_tensor(query_embedding):
            query_embedding = query_embedding.numpy()

        if self.index_type == "ivf_hnsw":
            self.index.nprobe = nprobe

        D, I = self.index.search(query_embedding, k)
        context = " ".join([self.documents[i] for i in I[0]])

        # Generate response using chat format
        prompt = f"""Based on the following context, please answer the question:

Context: {context}

Question: {query}

Answer: Let me help you with that."""

        response = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            echo=False
        )

        return response['choices'][0]['text'].strip()

def main():
    try:
        # Initialize RAPTOR system with GGUF model and HNSW index
        raptor = LlamaRaptorGGUF(
            model_path="models/phi-4-Q6_K.gguf",  # Update with your model path
            index_type="hnsw",
            hnsw_m=32,
            nlist=100
        )

        # Load documents
        import charset_normalizer

        with open("demo/sample.txt", "r", encoding="utf-8") as file:
            documents = file.readlines()

        
        # Add documents
        raptor.add_documents(documents)

        # Example query
        query = "What technologiy or methods are being talked about in this document"
        print("\nQuery:", query)

        # Generate answer
        answer = raptor.generate_answer(query)
        print("\nAnswer:", answer)

    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Clean up
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
