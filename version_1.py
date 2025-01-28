import logging
from llama_cpp import Llama
from typing import List, Dict, Optional
import pickle
import numpy as np

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

# --- Custom Embedding Model ---
class CustomEmbeddingModel:
    def __init__(self, model_path: str):
        self.model = Llama(model_path=model_path, embedding=True)

    def create_embedding(self, text: str) -> np.ndarray:
        response = self.model.create_embedding(text)
        return np.array(response["data"])  # Convert to NumPy array



# --- Custom Summarization Model ---
class CustomSummarizationModel:
    def __init__(self, model_path: str):
        self.model = Llama(model_path=model_path)

    def summarize(self, context: str, max_tokens: int = 150) -> str:
        response = self.model(prompt=f"Summarize: {context}", max_tokens=max_tokens)
        return response["choices"][0]["text"].strip()


# --- Custom Question-Answering Model ---
class CustomQAModel:
    def __init__(self, model_path: str):
        self.model = Llama(model_path=model_path)

    def answer_question(self, context: str, question: str) -> str:
        prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
        response = self.model(prompt=prompt, max_tokens=150)
        return response["choices"][0]["text"].strip()



# --- Hierarchical Node and Tree Structures ---
class Node:
    def __init__(self, text: str, index: int, children: Optional[List[int]] = None, embeddings: Optional[Dict[str, List[float]]] = None):
        self.text = text
        self.index = index
        self.children = children if children else []
        self.embeddings = embeddings if embeddings else {}


class Tree:
    def __init__(self, all_nodes: Dict[int, Node], root_nodes: List[Node], leaf_nodes: Dict[int, Node], num_layers: int):
        self.all_nodes = all_nodes
        self.root_nodes = root_nodes
        self.leaf_nodes = leaf_nodes
        self.num_layers = num_layers


# --- Retrieval-Augmentation Configuration ---
class RetrievalAugmentationConfig:
    def __init__(self, embedding_model, summarization_model, qa_model):
        self.embedding_model = embedding_model
        self.summarization_model = summarization_model
        self.qa_model = qa_model


# --- Retrieval-Augmentation System ---
class RetrievalAugmentation:
    def __init__(self, config: RetrievalAugmentationConfig, tree: Optional[Tree] = None):
        self.config = config
        self.tree = tree or Tree({}, [], {}, 0)
        self.max_context_window = 512  # Define maximum context window
        self.chunk_size = 256  # Smaller chunk size to leave room for queries

    def add_documents(self, text: str):
        chunks = self.split_text(text, max_tokens=self.chunk_size)
        logging.info(f"Split text into {len(chunks)} chunks.")
        
        for idx, chunk in enumerate(chunks):
            try:
                # Convert embedding to a simple list instead of numpy array
                embedding = self.config.embedding_model.create_embedding(chunk)
                if isinstance(embedding, np.ndarray):
                    embedding = embedding.tolist()
                
                node = Node(text=chunk, index=idx, embeddings={"custom_model": embedding})
                self.tree.all_nodes[idx] = node
                self.tree.leaf_nodes[idx] = node
            except Exception as e:
                logging.error(f"Error processing chunk {idx}: {e}")
                continue

        self.tree.root_nodes = list(self.tree.leaf_nodes.values())
        self.tree.num_layers = 1
        logging.info("Documents successfully added and tree created.")

    def split_text(self, text: str, max_tokens: int) -> List[str]:
        """
        Split text into smaller chunks while respecting sentence boundaries.
        """
        sentences = [s.strip() + "." for s in text.split(".") if s.strip()]
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            # Approximate token count as word count
            sentence_length = len(sentence.split())
            
            if current_length + sentence_length > max_tokens and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0
            
            current_chunk.append(sentence)
            current_length += sentence_length

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def euclidean_distance(self, vec1, vec2):
        """
        Calculate Euclidean distance between two vectors.
        """
        try:
            # Ensure inputs are lists or convert them
            if isinstance(vec1, dict):
                vec1 = list(vec1.values())
            if isinstance(vec2, dict):
                vec2 = list(vec2.values())
            
            if isinstance(vec1, np.ndarray):
                vec1 = vec1.tolist()
            if isinstance(vec2, np.ndarray):
                vec2 = vec2.tolist()
            
            if isinstance(vec1[0], (list, np.ndarray)):
                vec1 = vec1[0]
            if isinstance(vec2[0], (list, np.ndarray)):
                vec2 = vec2[0]
            
            arr1 = np.array(vec1, dtype=float)
            arr2 = np.array(vec2, dtype=float)
            
            return float(np.sqrt(np.sum((arr1 - arr2) ** 2)))
        except Exception as e:
            logging.error(f"Error calculating Euclidean distance: {e}")
            return float('inf')

    def retrieve_context(self, query: str, max_tokens: Optional[int] = None) -> str:
        """
        Retrieve relevant context while respecting token limits.
        """
        if max_tokens is None:
            max_tokens = self.chunk_size  # Use default chunk size if not specified
            
        try:
            query_embedding = self.config.embedding_model.create_embedding(query)
            
            # Calculate distances
            distances = []
            for node in self.tree.leaf_nodes.values():
                try:
                    node_embedding = node.embeddings["custom_model"]
                    distance = self.euclidean_distance(query_embedding, node_embedding)
                    distances.append((node.index, distance))
                except Exception as e:
                    logging.error(f"Error processing node {node.index}: {e}")
                    continue
            
            # Sort by distance
            distances.sort(key=lambda x: x[1])
            
            # Build context within token limit
            context_parts = []
            total_tokens = 0
            
            for idx, _ in distances:
                node = self.tree.leaf_nodes[idx]
                # Approximate token count
                tokens = len(node.text.split())
                
                if total_tokens + tokens > max_tokens:
                    break
                    
                context_parts.append(node.text)
                total_tokens += tokens
            
            return " ".join(context_parts).strip()
            
        except Exception as e:
            logging.error(f"Error in retrieve_context: {e}")
            return ""

    def answer_question(self, question: str) -> str:
        """
        Answer a question using retrieved context, respecting token limits.
        """
        try:
            # Use a smaller context size to leave room for the question and answer
            context = self.retrieve_context(question, max_tokens=self.chunk_size)
            
            # Construct a prompt that fits within context window
            prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
            
            # Get answer from QA model
            return self.config.qa_model.answer_question(context, question)
            
        except Exception as e:
            logging.error(f"Error in answer_question: {e}")
            return "Sorry, I encountered an error while processing your question."

    def save(self, path: str):
        with open(path, "wb") as file:
            pickle.dump(self.tree, file)
        logging.info(f"Tree saved to {path}.")

    def load(self, path: str):
        with open(path, "rb") as file:
            self.tree = pickle.load(file)
        logging.info(f"Tree loaded from {path}.")



# --- Main Execution ---
if __name__ == "__main__":
    # Initialize custom models
    embedding_model = CustomEmbeddingModel("models/llama-2-13b.Q4_K_M.gguf")
    summarization_model = CustomSummarizationModel("models/llama-2-13b.Q4_K_M.gguf")
    qa_model = CustomQAModel("models/llama-2-13b.Q4_K_M.gguf")

    # Configure and initialize the system
    config = RetrievalAugmentationConfig(embedding_model, summarization_model, qa_model)
    RA = RetrievalAugmentation(config)

    # Add documents
    with open("demo/sample.txt", "r") as file:
        text = file.read()
    RA.add_documents(text)

    # Answer a question
    question = "What is the significance of recursive processing?"
    answer = RA.answer_question(question)
    print("Answer:", answer)

    # Save and load the tree
    RA.save("custom_tree.pkl")
