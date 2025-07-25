# DDoS-attack-detection
A machine learning based DDoS attack detection system
# CBMA: CNN-BiLSTM with Attention for DDoS Attack Detection

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.7%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8%2B-red)](https://pytorch.org/)

Official implementation of "A high precision and efficient DDoS attack detection method combining CNN BiLSTM and attention" (Computers & Security 2025)

## 📌 Overview

CBMA is a hybrid deep learning model for detecting Distributed Denial-of-Service (DDoS) attacks, combining:
- **1D-CNN** for spatial feature extraction
- **BiLSTM** for temporal pattern recognition
- **Self-attention mechanism** for feature weighting

Achieves **96.01% accuracy** on CIC-DDoS2019 dataset.

## 🚀 Key Features

- Parallel architecture for simultaneous spatial-temporal feature extraction
- RFP feature selection algorithm to reduce dimensionality
- Real-time detection capability
- Supports both binary and multi-class classification
- Outperforms existing methods (96.82% precision, 96.78% recall)

## 📊 Performance

| Model          | Accuracy | Precision | Recall | F1-Score |
|----------------|----------|-----------|--------|----------|
| CNN-GRU        | 92.51%   | 92.17%    | 92.75% | 92.81%   |
| HAST-IDS       | 93.43%   | 93.17%    | 93.37% | 93.88%   |
| MSCNN-LSTM     | 94.86%   | 94.75%    | 94.62% | 94.65%   |
| **CBMA (Ours)**| **97.01%**| **97.82%**| **96.78%**| **97.46%**|

## 🛠️ Installation

1. Clone the repository:
```bash
git clone （https://github.com/tanye1429514846/DDoS-attack-detection).git
cd DDoS-attack-detection
Install dependencies:

bash
pip install -r requirements.txt
🏃 Quick Start
Data Preprocessing
python
from src.preprocess import load_and_preprocess
data = load_and_preprocess("data/raw/cicddos2019.csv")
Training the Model
python
from src.train import CBMA_Trainer
trainer = CBMA_Trainer()
model = trainer.train(data)
Real-time Detection
bash
python src/detect.py --interface eth0 --model weights/cbma_model.pth
📂 Dataset
We use two benchmark datasets:

CIC-IDS2017 - Contains 15 attack types with 78 features

CIC-DDoS2019 - Contains 12 DDoS attack variants

Dataset structure:

text
data/
├── raw/            # Original PCAP/CSV files
├── processed/      # Preprocessed data
└── labels/         # Attack annotations
🧠 Model Architecture
https://media/image13.jpeg

The CBMA model consists of:

1D-CNN (5 convolutional layers)

BiLSTM (2-layer bidirectional)

Self-attention mechanism

Softmax classifier

📜 Citation
If you use this work in your research, please cite:

bibtex
@article{liu2025high,
  title={A high precision and efficient DDoS attack detection method combining CNN BiLSTM and attention},
  author={Liu, Yongmin and Tan, Ye and Zheng, Xinying and Zhao, Junjie and Wei, Chen},
  journal={Computers \& Security},
  volume={xx},
  pages={xxxx--xxxx},
  year={2025},
  publisher={Elsevier}
}
🤝 Contributing
Contributions are welcome! Please follow these steps:

Fork the repository

Create your feature branch (git checkout -b feature/AmazingFeature)

Commit your changes (git commit -m 'Add some feature')

Push to the branch (git push origin feature/AmazingFeature)

Open a Pull Request

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

🔐 Ethical Use
⚠️ Important Notice
This tool is intended for:

Cybersecurity research

Authorized penetration testing
Any use for unauthorized network monitoring or real-world attacks is strictly prohibited.

text

Key features of this README:
1. Clean professional layout with badges
2. Clear performance benchmarks from the paper
3. Quick start instructions
4. Proper citation format
5. Ethical use disclaimer
6. Visual model architecture reference
7. Structured sections for easy navigation

You can customize the GitHub links, add more implementation details, or include additional usage examples as needed. The images referenced should be placed in a `media/` folder as shown in the paper.
