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
```

### Step 2: Create and Activate a Virtual Environment
It is highly recommended to use a virtual environment to manage dependencies.

On Windows:
```
python -m venv venv
venv\Scripts\activate
```
On macOS/Linux:
```
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
Install all required packages using the requirements.txt file. This project requires a GPU-enabled version of PyTorch.
```
pip install -r requirements.txt
```

## 3. How to Run the Experiment
The entire workflow is contained in the "Assignment 3.py" script. To run all four training and evaluation experiments, execute the following command:

```
python "Assignment 3.py"
```
The script will automatically:
1. Download the imdb dataset and pretrained models from the Hugging Face Hub.
2. Train each of the four models sequentially.
3. Resume training from the latest checkpoint if a run is interrupted.
4. Print a final comparison table to the console.
5. Save the final numerical results to final_results.csv.
 6. Generate and save two plots:
    * results_comparison.png: A bar chart comparing the final accuracy and F1 scores.
    * all_training_histories.png: A 2x2 grid showing the training and validation loss curves for all four models.
    * All model checkpoints and outputs will be saved in the ./results/ directory.
