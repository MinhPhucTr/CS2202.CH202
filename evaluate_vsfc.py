"""
Evaluate ML-Embed-0.6B on a Vietnamese text classification dataset.
Loads train.parquet and test.parquet directly from Hugging Face Hub.
"""

import os
import torch
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# DATASET LOADER: HUGGING FACE HUB (PARQUET)
def load_vietnamese_dataset():
    # REPLACE with your actual Hugging Face repository ID
    repo_id = "phuc98080/CS2202Dataset"
    print(f"[INFO] Loading parquet dataset from Hugging Face Hub: {repo_id}...")

    # Load train.parquet and test.parquet directly from the repository
    dataset = load_dataset(
        repo_id,
        data_files={"train": "train.parquet", "test": "test.parquet"}
    )

    print("[INFO] Successfully loaded dataset from Hugging Face Hub!")
    return dataset["train"], dataset["test"]


def main():
    # 1. Configure hardware device (GPU/CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device.upper()}")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 2. Load ML-Embed-0.6B model
    model_name = "codefuse-ai/ML-Embed-0.6B"
    print(f"[INFO] Loading model: {model_name}...")

    model_kwargs = {"torch_dtype": torch.bfloat16} if device == "cuda" else {}
    model = SentenceTransformer(model_name, model_kwargs=model_kwargs, device=device)
    model.max_seq_length = 512

    # 3. Load dataset from Hugging Face
    train_data, test_data = load_vietnamese_dataset()

    # 4. Extract sentence embeddings (Change "sentence" to "text" if needed)
    print("[INFO] Encoding training sentences...")
    with torch.inference_mode():
        X_train = model.encode(
            train_data["sentence"],
            batch_size=16,
            show_progress_bar=True,
            normalize_embeddings=True
        )
        y_train = np.array(train_data["label"])

    print("[INFO] Encoding test sentences...")
    with torch.inference_mode():
        X_test = model.encode(
            test_data["sentence"],
            batch_size=16,
            show_progress_bar=True,
            normalize_embeddings=True
        )
        y_test = np.array(test_data["label"])

    # 5. Train Linear Classifier & Evaluate
    print("[INFO] Training Logistic Regression classifier...")
    classifier = LogisticRegression(max_iter=1000, C=1.0)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred) * 100

    # 6. Display evaluation results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS: ML-Embed-0.6B:")
    print("=" * 60)
    print(f">> ACCURACY SCORE: {accuracy:.2f}%")
    print("-" * 60)
    label_names = ["Negative", "Neutral", "Positive"]
    print(classification_report(y_test, y_pred, target_names=label_names))
    print("=" * 60)


if __name__ == "__main__":
    main()