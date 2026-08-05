# CS2202.CH202
# Evaluating ML-Embed-0.6B on Vietnamese Datasets

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository provides a streamlined pipeline for evaluating the **`codefuse-ai/ML-Embed-0.6B`** embedding model on Vietnamese text classification tasks (such as Sentiment Analysis and Topic Classification), following standard MTEB evaluation protocols.

---

## 📂 Project Structure

```text
├── data/
│   ├── train.csv         # Training dataset (e.g., UIT-VSFC or VNTC)
│   └── test.csv          # Test dataset
├── evaluate_vsfc.py      # Main evaluation script for classification
├── requirements.txt      # Required Python packages
└── README.md             # Project documentation
⚙️ Prerequisites & Installation1. Clone the RepositoryBashgit clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name
2. Install DependenciesFor NVIDIA GPU Users (Recommended for RTX 3000/4000 series):If you are using an NVIDIA GPU (e.g., RTX 4060), install PyTorch with CUDA support first:Bashpip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu124](https://download.pytorch.org/whl/cu124)
pip install -r requirements.txt
(Tip: Verify your GPU status by running: python -c "import torch; print(torch.cuda.is_available())")For CPU-only or macOS Users:Bashpip install -r requirements.txt
🚀 Running the EvaluationEnsure your dataset files (train.csv and test.csv containing text/sentence and label columns) are properly placed inside the data/ directory.Execute the evaluation script:Bashpython evaluate_vsfc.py
The script will automatically:Detect and utilize your hardware device (CUDA or CPU).  Load the codefuse-ai/ML-Embed-0.6B model using optimized precision (bfloat16 on CUDA).  Encode text sentences into dense embeddings with gradient inference disabled.  Train a standard Logistic Regression classifier and print out comprehensive metrics (Accuracy, Precision, Recall, F1-score)[cite: 3].📊 Evaluation MethodologyFollowing standard embedding evaluation guidelines:Feature Extraction: Sentences are mapped into normalized dense vectors using SentenceTransformer.Linear Probing: A simple Logistic Regression model (scikit-learn) is trained on top of the fixed embeddings to measure semantic linear separability without modifying model weights.
