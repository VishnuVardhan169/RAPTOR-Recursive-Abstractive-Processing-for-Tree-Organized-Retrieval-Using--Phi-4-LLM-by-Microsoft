import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
from version_4 import LlamaRaptorGGUF  # Import your model class or methods directly

# Function to calculate number of tokens in context
def calculate_tokens(contexts, model):
    token_counts = [len(model.llm.tokenize(context)) for context in contexts]
    return token_counts

# Function to visualize token counts
def visualize_tokens(token_counts):
    plt.figure(figsize=(10, 6))
    plt.hist(token_counts, bins=20, color='skyblue', edgecolor='black')
    plt.xlabel("Number of Tokens")
    plt.ylabel("Frequency")
    plt.title("Distribution of Tokens in Contexts")
    plt.show()

# Function to evaluate and visualize accuracy
def evaluate_accuracy(model, queries, true_answers):
    predictions = [model.generate_answer(query) for query in queries]
    accuracies = [accuracy_score([true], [pred]) for true, pred in zip(true_answers, predictions)]

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(queries)), accuracies, color='lightgreen', edgecolor='black')
    plt.xlabel("Query Index")
    plt.ylabel("Accuracy")
    plt.title("Accuracy of LLM Responses")
    plt.show()

    return predictions, accuracies

# Function to perform context analysis
def context_analysis(contexts):
    embedding_model = SentenceTransformer('bert-base-nli-mean-tokens')
    embeddings = embedding_model.encode(contexts)

    # Analyze embedding similarities (example)
    similarity_matrix = np.inner(embeddings, embeddings)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(similarity_matrix, cmap='viridis', interpolation='nearest')
    plt.colorbar(label="Cosine Similarity")
    plt.title("Context Similarity Matrix")
    plt.show()

# Main function to initialize model and perform visualizations
if __name__ == "__main__":
    try:
        # Initialize the RAPTOR model
        raptor_model = LlamaRaptorGGUF(
            model_path="models/phi-4-Q6_K.gguf",  # Update with the correct model path
            n_ctx=2048,
            n_gpu_layers=35,
            index_type="hnsw",
            hnsw_m=32,
            nlist=100,
            use_gpu=True
        )

        # Load data from sample.txt
        with open("demo/sample.txt", "r", encoding="utf-8") as file:
            example_contexts = [line.strip() for line in file.readlines() if line.strip()]

        # Example queries based on the document
        example_queries = [
            "What is RAPTOR?",
            "Explain the objectives of RAPTOR.",
            "How does RAPTOR perform tree construction?"
        ]

        # Ground truth answers for evaluation (replace with accurate answers if available)
        true_answers = [
            "Recursive Abstractive Processing for Tree-Organized Retrieval.",
            "Text chunking, embedding, clustering, summarization, tree construction, and evaluation.",
            "Organize summaries into a tree structure with nodes containing summaries and links to details."
        ]

        # Calculate and visualize token counts
        token_counts = calculate_tokens(example_contexts, raptor_model)
        visualize_tokens(token_counts)

        # Evaluate accuracy and visualize
        predictions, accuracies = evaluate_accuracy(raptor_model, example_queries, true_answers)

        # Perform context analysis
        context_analysis(example_contexts)

    except Exception as e:
        print(f"Error: {e}")

    finally:
        # Ensure proper cleanup of the model
        raptor_model.llm.close()
