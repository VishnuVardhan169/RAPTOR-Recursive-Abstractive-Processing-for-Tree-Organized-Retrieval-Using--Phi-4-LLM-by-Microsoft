from ctransformers import AutoModelForCausalLM
import torch
from raptor import RetrievalAugmentation, RetrievalAugmentationConfig, BaseSummarizationModel, BaseQAModel, BaseEmbeddingModel
import os

class ClaudeSummarizationModel(BaseSummarizationModel):
    def __init__(self, model_path="D:/Raptor_modified_master/models/claude2-alpaca-13b.Q6_K.gguf"):
        # Load Claude for summarization
        self.model = AutoModelForCausalLM.from_pretrained(model_path, model_type="llama")

    def summarize(self, context, max_tokens=150):
        prompt = f"Summarize the following:\n{context}\nSummary:"
        return self.model(prompt, max_new_tokens=max_tokens)

class LlamaQAModel(BaseQAModel):
    def __init__(self, model_path="D:/Raptor_modified_master/models/llama-2-13b.Q4_K_M.gguf"):
        # Load Llama for question answering
        self.model = AutoModelForCausalLM.from_pretrained(model_path, model_type="llama")

    def answer_question(self, context, question):
        prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
        return self.model(prompt, max_new_tokens=150)

class CustomEmbeddingModel(BaseEmbeddingModel):
    def __init__(self):
        pass

    def create_embedding(self, text):
        # Placeholder for embedding logic
        return [0.0] * 768  # Dummy embedding

if __name__ == "__main__":
    # Check if ROCm is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize custom models
    summarization_model = ClaudeSummarizationModel()
    qa_model = LlamaQAModel()
    embedding_model = CustomEmbeddingModel()

    # Create RAPTOR configuration
    config = RetrievalAugmentationConfig(
        summarization_model=summarization_model,
        qa_model=qa_model,
        embedding_model=embedding_model
    )
    # Initialize RAPTOR
    RA = RetrievalAugmentation(config=config)

    # Example usage
    with open("D:/Raptor_modified_master/demo/sample.txt", "r", encoding="utf-8") as file:
        text = file.read()

    RA.add_documents(text)

    question = "What is the title of this document?"
    answer = RA.answer_question(question=question)
    print("Answer:", answer)

    # Save the tree
    SAVE_PATH = "D:/Raptor_modified_master/saved_tree"
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    RA.save(SAVE_PATH)
    print(f"Tree saved at {SAVE_PATH}")
