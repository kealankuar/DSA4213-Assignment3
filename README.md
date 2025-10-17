# Assignment 3: Efficient Adaptation of Pretrained Transformers via Knowledge Distillation and PEFT

This repository contains the code and results for DSA4213 Assignment 3. The project implements and analyzes four different fine-tuning strategies to adapt pretrained Transformer models for sentiment analysis, with a focus on comparing full fine-tuning against parameter-efficient methods and knowledge distillation.

## 1. Project Overview

This project explores the trade-offs between model performance and computational efficiency in fine-tuning. A creative experiment was designed to go beyond the assignment's minimum requirements by synthesizing concepts from multiple lectures.

* **Task**: Text Classification (Sentiment Analysis)
* **Dataset**: `imdb`
* **Models Compared**:
    1.  **Teacher Model**: `bert-base-uncased` (fully fine-tuned).
    2.  **Baseline Student**: `distilbert-base-uncased` (fully fine-tuned).
    3.  **LoRA Student**: `distilbert-base-uncased` fine-tuned using Low-Rank Adaptation (LoRA).
    4.  **Distil-LoRA Student**: `distilbert-base-uncased` fine-tuned using LoRA and Knowledge Distillation from the teacher model.

This experiment combines concepts from **Lecture 8 (Efficient Adaptation)** and **Lecture 9 (Knowledge Distillation)** to provide a comprehensive analysis.

## 2. Setup and Installation

Follow these steps to set up the environment and run the code. A Python version of 3.9 or later and an NVIDIA GPU are required.

### Step 1: Clone the Repository
```bash
git clone https://github.com/kealankuar/DSA4213-Assignment3.git
cd DSA4213-Assignment3