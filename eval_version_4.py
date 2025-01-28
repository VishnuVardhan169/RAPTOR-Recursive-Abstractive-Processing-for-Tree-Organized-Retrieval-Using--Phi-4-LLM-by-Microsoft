import torch
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from datasets import load_dataset

# Load Qasper and NarrativeQA datasets
def load_datasets():
    print("Loading Qasper and NarrativeQA datasets...")
    qasper = load_dataset("qasper", trust_remote_code=True)
    narrative_qa = load_dataset("narrativeqa")
    return qasper, narrative_qa

# Preprocess Qasper dataset
def preprocess_qasper(qasper):
    print("Preprocessing Qasper dataset...")
    data = qasper["train"]
    X, y = [], []
    for item in data:
        question = item["question"]
        answer = item["answers"]
        if answer:
            X.append(question)
            y.append(answer[0]["text"] if len(answer[0]["text"]) > 0 else "")
    return X, y

# Preprocess NarrativeQA dataset
def preprocess_narrative_qa(narrative_qa):
    print("Preprocessing NarrativeQA dataset...")
    data = narrative_qa["train"]
    X, y = [], []
    for item in data:
        question = item["question"]
        answer = item["answer1"]
        if question and answer:
            X.append(question)
            y.append(answer)
    return X, y


# Function to load the model
def load_model(file_path="raptor_model.pkl"):
    print(f"Loading model from {file_path}...")
    with open(file_path, "rb") as file:
        model = pickle.load(file)
    print("Model loaded successfully.")
    return model

# Function to evaluate the model's accuracy
def evaluate_model(model, datasets):
    results = {}
    for name, (X_test, y_true) in datasets.items():
        print(f"Evaluating on {name}...")
        y_pred = [model.generate_answer(query) for query in X_test]  # Replace with actual prediction method
        accuracy = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, zero_division=1)
        results[name] = {
            "accuracy": accuracy,
            "report": report
        }
    return results

# Function to visualize model performance
def plot_accuracies(results):
    dataset_names = list(results.keys())
    accuracies = [results[name]["accuracy"] for name in dataset_names]

    plt.figure(figsize=(10, 6))
    plt.bar(dataset_names, accuracies, color='skyblue')
    plt.xlabel("Datasets")
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy Across Datasets")
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Main function to load data, preprocess, evaluate, and save model
if __name__ == "__main__":
    # Load datasets
    qasper, narrative_qa = load_datasets()

    # Preprocess datasets
    X_qasper, y_qasper = preprocess_qasper(qasper)
    X_narrative, y_narrative = preprocess_narrative_qa(narrative_qa)

    # Prepare datasets for evaluation
    datasets = {
        "Qasper": (X_qasper, y_qasper),
        "NarrativeQA": (X_narrative, y_narrative)
    }

    # Initialize RAPTOR model (replace with your actual model initialization)
    # raptor_model = LlamaRaptorGGUF(...)  # Uncomment and configure your RAPTOR model

    # Save the model
    # save_model(raptor_model)  # Uncomment after defining raptor_model

    # Load the model (for future use)
    # raptor_model = load_model()  # Uncomment to load the saved model

    # Evaluate model
    # results = evaluate_model(raptor_model, datasets)  # Uncomment after defining raptor_model

    # Visualize results
    # plot_accuracies(results)  # Uncomment to visualize results
