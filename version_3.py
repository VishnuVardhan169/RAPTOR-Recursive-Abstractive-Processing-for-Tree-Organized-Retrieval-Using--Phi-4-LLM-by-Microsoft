# Install the required dependencies:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6
# pip install faiss-cpu transformers datasets

import faiss
import numpy as np
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import torch
import os

# Check if ROCm (AMD GPU) is available
if torch.backends.cuda.is_built():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Using AMD GPU: {torch.cuda.get_device_name(0)}")
        print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    device = torch.device("cpu")
    print("No GPU detected, using CPU only")

# Step 1: Load and Process Text
file_path = 'demo/sample.txt'

try:
    with open(file_path, 'r') as file:
        full_text = file.read()
    document_chunks = full_text.split('\n\n')
    print(f"Loaded {len(document_chunks)} document chunks")

except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
    document_chunks = []
except Exception as e:
    print(f"Error reading file: {e}")
    document_chunks = []

# Step 2: FAISS for Document Retrieval
# Using CPU for FAISS as it's often more efficient for smaller datasets
dim = 768
embeddings = np.random.rand(len(document_chunks), dim).astype('float32')
query_embedding = np.random.rand(dim).astype('float32')

index = faiss.IndexFlatL2(dim)
index.add(embeddings)

k = 3
D, I = index.search(query_embedding.reshape(1, -1), k)
top_k_results = [document_chunks[i] for i in I[0]]

# Step 3: Fine-Tuning T5 with AMD GPU Support
model_name = 't5-large'
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Move model to GPU if available
model = model.to(device)

# Load dataset
dataset = load_dataset('squad')

def tokenize_function(examples):
    questions = examples['question']
    answers = [answer['text'][0] if answer['text'] else "" for answer in examples['answers']]
    
    # Prepare the input format T5 expects
    inputs = [f"question: {q}  context: {c}" for q, c in zip(questions, examples['context'])]
    
    model_inputs = tokenizer(
        inputs,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    
    # Tokenize targets (answers)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            answers,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Tokenize the dataset
tokenized_datasets = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset["train"].column_names
)

# Configure training arguments with GPU optimization
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,  # Adjust based on your GPU memory
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    fp16=True,  # Enable mixed precision training
    gradient_accumulation_steps=2,  # Helps with memory management
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
)

# Train the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained('./fine_tuned_model')
tokenizer.save_pretrained('./fine_tuned_model')

# Step 4: Answer Generation
def generate_answer(question, context, max_length=150):
    input_text = f"question: {question}  context: {context}"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(device)
    
    # Generate answer with GPU acceleration
    outputs = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        early_stopping=True
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example usage
context = " ".join(top_k_results)
user_question = "Who is the author and supervisor here and also give me raptor objectives "
answer = generate_answer(user_question, context)
print(f"Generated Answer: {answer}")

# Clean up GPU memory
if device.type == "cuda":
    torch.cuda.empty_cache()