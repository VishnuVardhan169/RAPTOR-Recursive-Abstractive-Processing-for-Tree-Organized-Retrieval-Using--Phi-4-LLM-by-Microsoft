import logging
import pickle
from llama_cpp import Llama
from typing import List, Dict, Optional
import numpy as np

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

# --- Custom Embedding Model ---
class CustomEmbeddingModel:
    def __init__(self, model_path: str):
        self.model = Llama(model_path=model_path, embedding=True)

    def create_embedding(self, text: str) -> np.ndarray:
        response = self.model.create_embedding(text)
        return np.array(response["data"])  # Convert to NumPy array


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
    def __init__(self, embedding_model, qa_model):
        self.embedding_model = embedding_model
        self.qa_model = qa_model


# --- Retrieval-Augmentation System ---
class RetrievalAugmentation:
    def __init__(self, config: RetrievalAugmentationConfig, tree: Optional[Tree] = None):
        self.config = config
        self.tree = tree
        self.chunk_size = 256  # Smaller chunk size to leave room for queries

    def euclidean_distance(self, vec1, vec2):
        """
        Calculate Euclidean distance between two vectors.
        """
        try:
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


# --- Main Execution ---
if __name__ == "__main__":
    # Initialize custom models
    embedding_model = CustomEmbeddingModel("models/llama-2-13b.Q4_K_M.gguf")
    qa_model = CustomQAModel("models/llama-2-13b.Q4_K_M.gguf")

    # Load the previously saved tree
    try:
        with open("custom_tree.pkl", "rb") as file:
            tree = pickle.load(file)
            logging.info("Tree loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading tree: {e}")
        tree = None

    # Check if the tree is loaded correctly
    if tree:
        # Configure and initialize the system
        config = RetrievalAugmentationConfig(embedding_model, qa_model)
        RA = RetrievalAugmentation(config, tree)

        # Ask questions
        while True:
            question = input("Enter your question (or type 'exit' to quit): ")
            if question.lower() == 'exit':
                break

            answer = RA.answer_question(question)
            print("Answer:", answer)
    else:
        print("Failed to load the tree. Ensure the custom_tree.pkl file exists.")
