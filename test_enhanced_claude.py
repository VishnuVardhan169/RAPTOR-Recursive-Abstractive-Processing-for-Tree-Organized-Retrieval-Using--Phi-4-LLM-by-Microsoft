from ctransformers import AutoModelForCausalLM
from raptor import BaseQAModel

class EnhancedClaudeQAModel(BaseQAModel):
    def __init__(self, model_path="models/claude2-alpaca-13b.Q6_K.gguf"):
        # Load model using ctransformers
        self.model = AutoModelForCausalLM.from_pretrained(model_path, model_type="llama")

    def answer_question(self, context, question):
        prompt = f"""Context: {context}
        Question: {question}
        Instructions: Provide a concise and accurate answer based only on the given context."""
        answer = self.model(prompt, max_new_tokens=150)
        return answer

if __name__ == "__main__":
    # Instantiate the model
    qa_model = EnhancedClaudeQAModel(model_path="models/claude2-alpaca-13b.Q6_K.gguf")

    # Test the model
    context = "Claude 2 is a highly advanced language model designed for various AI applications."
    question = "What is Recurssive abstractive preprocessing for tress organized retreival"
    answer = qa_model.answer_question(context, question)
    
    print("Answer:", answer)
