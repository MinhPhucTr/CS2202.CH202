"""
Evaluate ML-Embed-0.6B on the Vietnamese News Topic Classification Dataset (VNTC / VN-News-10).
Loads vntc_train.parquet and vntc_test.parquet from Hugging Face Hub and subsamples data.
"""

import torch
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# DATASET LOADER: HUGGING FACE HUB WITH SUBSAMPLING
def load_vntc_dataset():
    # REPLACE with your actual Hugging Face repository ID
    repo_id = "phuc98080/CS2202Dataset"
    print(f"[INFO] Loading dataset from Hugging Face Hub: {repo_id}...")

    dataset = load_dataset(
        repo_id,
        data_files={"train": "vntc_train.parquet", "test": "vntc_test.parquet"}
    )

    # Shuffle with a fixed seed to ensure balanced topics, then select subsets
    print("[INFO] Subsampling -> Train: 1900 rows | Test: 2500 rows...")
    train_subset = dataset["train"].shuffle(seed=42).select(range(1900))
    test_subset = dataset["test"].shuffle(seed=42).select(range(2500))

    print("[INFO] Successfully loaded and subsampled VNTC dataset!")
    return train_subset, test_subset


def main():
    # 1. Configure hardware device
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

    # 3. Load subsampled VNTC dataset
    train_data, test_data = load_vntc_dataset()


    # 4. Extract sentence embeddings
    print("[INFO] Encoding training articles...")
    with torch.inference_mode():
        X_train = model.encode(
            train_data["text"],
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True
        )
        y_train = np.array(train_data["label"])

    print("[INFO] Encoding test articles...")
    with torch.inference_mode():
        X_test = model.encode(
            test_data["text"],
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True
        )
        y_test = np.array(test_data["label"])

    # 5. Train Linear Classifier & Evaluate
    print("[INFO] Training Logistic Regression classifier on VN-News-10...")
    classifier = LogisticRegression(max_iter=1000, C=1.0)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred) * 100

    # 6. Display evaluation results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS: ML-Embed-0.6B")
    print("=" * 60)
    print(f">> ACCURACY SCORE: {accuracy:.2f}%")
    print("-" * 60)

    # VNTC standard 10 topic names
    topic_names = [
        "Chinh tri Xa hoi", "Doi song", "Khoa hoc", "Kinh doanh", "Phap luat",
        "Suc khoe", "The gioi", "The thao", "Van hoa", "Vi tinh"
    ]
    print(classification_report(y_test, y_pred, target_names=topic_names))
    print("=" * 60)


if __name__ == "__main__":
    main()