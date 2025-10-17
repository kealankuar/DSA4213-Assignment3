import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from evaluate import load
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig, get_peft_model, TaskType

# ==============================================================================
# 1. SETUP: MODELS AND DATASET
# ==============================================================================
teacher_checkpoint = "bert-base-uncased"
student_checkpoint = "distilbert-base-uncased"
dataset_name = "imdb"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ==============================================================================
# 2. DATA LOADING AND PREPROCESSING
# ==============================================================================
print("Loading and preprocessing the dataset...")
dataset = load_dataset(dataset_name)

tokenizer_teacher = AutoTokenizer.from_pretrained(teacher_checkpoint, use_fast=True)
tokenizer_student = AutoTokenizer.from_pretrained(student_checkpoint, use_fast=True)

def preprocess_function(tokenizer, max_len):
    def fn(examples):
        return tokenizer(examples["text"], truncation=True, padding=True, max_length=max_len)
    return fn

train_sample_size = 1000
eval_sample_size = 200
max_seq_length = 256

tokenized_datasets_teacher = dataset.map(preprocess_function(tokenizer_teacher, max_seq_length), batched=True)
tokenized_datasets_student = dataset.map(preprocess_function(tokenizer_student, max_seq_length), batched=True)

train_dataset_teacher = tokenized_datasets_teacher["train"].shuffle(seed=42).select(range(train_sample_size))
eval_dataset_teacher = tokenized_datasets_teacher["test"].shuffle(seed=42).select(range(eval_sample_size))
train_dataset_student = tokenized_datasets_student["train"].shuffle(seed=42).select(range(train_sample_size))
eval_dataset_student = tokenized_datasets_student["test"].shuffle(seed=42).select(range(eval_sample_size))

data_collator_teacher = DataCollatorWithPadding(tokenizer=tokenizer_teacher)
data_collator_student = DataCollatorWithPadding(tokenizer=tokenizer_student)

# ==============================================================================
# 3. EVALUATION METRICS
# ==============================================================================
accuracy_metric = load("accuracy")
f1_metric = load("f1")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    acc = accuracy_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="weighted")
    return {"accuracy": acc["accuracy"], "f1": f1["f1"]}
    
all_results = {}
all_trainers = {}

# ==============================================================================
# 4. TRAIN/LOAD THE TEACHER MODEL (BERT-base, Full FT)
# ==============================================================================
print("\n--- Phase 1: Teacher Model (BERT-base Full Fine-Tuning) ---")
output_dir_teacher = "./results/teacher_bert_base"
training_args_teacher = TrainingArguments(
    output_dir=output_dir_teacher, learning_rate=2e-5, per_device_train_batch_size=4,
    per_device_eval_batch_size=4, gradient_accumulation_steps=4, fp16=True, num_train_epochs=3,
    weight_decay=0.01, eval_strategy="epoch", save_strategy="epoch", logging_strategy="steps",
    logging_steps=50, load_best_model_at_end=True,
)
teacher_model = AutoModelForSequenceClassification.from_pretrained(teacher_checkpoint, num_labels=2).to(device)
teacher_trainer = Trainer(
    model=teacher_model, args=training_args_teacher, train_dataset=train_dataset_teacher,
    eval_dataset=eval_dataset_teacher, tokenizer=tokenizer_teacher, data_collator=data_collator_teacher,
    compute_metrics=compute_metrics,
)
last_checkpoint_teacher = get_last_checkpoint(output_dir_teacher)
teacher_trainer.train(resume_from_checkpoint=last_checkpoint_teacher)
eval_results_teacher = teacher_trainer.evaluate()
all_results['Teacher (BERT-base, Full FT)'] = eval_results_teacher
all_trainers['Teacher (BERT-base, Full FT)'] = teacher_trainer

# ==============================================================================
# 5. BASELINE STUDENT 1: DistilBERT, Full FT
# ==============================================================================
print("\n--- Phase 2: Baseline Student (DistilBERT Full Fine-Tuning) ---")
output_dir_student_full = "./results/student_full_ft"
training_args_student = TrainingArguments(
    output_dir=output_dir_student_full, learning_rate=2e-5, per_device_train_batch_size=4,
    per_device_eval_batch_size=4, gradient_accumulation_steps=4, fp16=True, num_train_epochs=3,
    weight_decay=0.01, eval_strategy="epoch", save_strategy="epoch", logging_strategy="steps",
    logging_steps=50, load_best_model_at_end=True,
)
student_model_full = AutoModelForSequenceClassification.from_pretrained(student_checkpoint, num_labels=2).to(device)
student_trainer_full = Trainer(
    model=student_model_full, args=training_args_student, train_dataset=train_dataset_student,
    eval_dataset=eval_dataset_student, tokenizer=tokenizer_student, data_collator=data_collator_student,
    compute_metrics=compute_metrics,
)
last_checkpoint_student_full = get_last_checkpoint(output_dir_student_full)
student_trainer_full.train(resume_from_checkpoint=last_checkpoint_student_full)
eval_results_student_full = student_trainer_full.evaluate()
all_results['Student (DistilBERT, Full FT)'] = eval_results_student_full
all_trainers['Student (DistilBERT, Full FT)'] = student_trainer_full

# ==============================================================================
# 6. BASELINE STUDENT 2: DistilBERT, LoRA only
# ==============================================================================
print("\n--- Phase 3: Baseline LoRA Student (No Distillation) ---")
output_dir_student_lora = "./results/student_lora"
training_args_student.output_dir = output_dir_student_lora
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS, r=8, lora_alpha=16, lora_dropout=0.1, bias="none",
    target_modules=["q_lin", "v_lin"],
)
student_model_lora = AutoModelForSequenceClassification.from_pretrained(student_checkpoint, num_labels=2).to(device)
student_model_lora = get_peft_model(student_model_lora, lora_config)
student_trainer_lora = Trainer(
    model=student_model_lora, args=training_args_student, train_dataset=train_dataset_student,
    eval_dataset=eval_dataset_student, tokenizer=tokenizer_student, data_collator=data_collator_student,
    compute_metrics=compute_metrics,
)
last_checkpoint_student_lora = get_last_checkpoint(output_dir_student_lora)
student_trainer_lora.train(resume_from_checkpoint=last_checkpoint_student_lora)
eval_results_student_lora = student_trainer_lora.evaluate()
all_results['Student (DistilBERT, LoRA)'] = eval_results_student_lora
all_trainers['Student (DistilBERT, LoRA)'] = student_trainer_lora

# ==============================================================================
# 7. ADVANCED EXPERIMENT: DistilBERT, Distil-LoRA
# ==============================================================================
print("\n--- Phase 4: Knowledge Distillation with LoRA (Distil-LoRA) ---")
output_dir_distil_lora = "./results/distil_lora_student"
training_args_student.output_dir = output_dir_distil_lora

class DistillationTrainer(Trainer):
    def __init__(self, *args, teacher_model=None, temperature=2.0, alpha=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.temperature = temperature
        self.alpha = alpha

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs_student = model(**inputs)
        loss_student_ce = outputs_student.loss
        teacher_inputs = {k: v.to(self.teacher_model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs_teacher = self.teacher_model(**teacher_inputs)
        loss_distill = nn.KLDivLoss(reduction="batchmean")(
            F.log_softmax(outputs_student.logits / self.temperature, dim=-1),
            F.softmax(outputs_teacher.logits / self.temperature, dim=-1)
        ) * (self.temperature ** 2)
        loss = self.alpha * loss_student_ce + (1 - self.alpha) * loss_distill
        return (loss, outputs_student) if return_outputs else loss

student_model_distil = AutoModelForSequenceClassification.from_pretrained(student_checkpoint, num_labels=2).to(device)
student_model_distil = get_peft_model(student_model_distil, lora_config)
teacher_model.eval()

distil_trainer = DistillationTrainer(
    model=student_model_distil, teacher_model=teacher_model, args=training_args_student,
    train_dataset=train_dataset_student, eval_dataset=eval_dataset_student,
    tokenizer=tokenizer_student, data_collator=data_collator_student, compute_metrics=compute_metrics,
    temperature=5.0, alpha=0.25,
)
last_checkpoint_distil_lora = get_last_checkpoint(output_dir_distil_lora)
distil_trainer.train(resume_from_checkpoint=last_checkpoint_distil_lora)
eval_results_distil_lora = distil_trainer.evaluate()
all_results['Student (DistilBERT, Distil-LoRA)'] = eval_results_distil_lora
all_trainers['Student (DistilBERT, Distil-LoRA)'] = distil_trainer

# ==============================================================================
# 8. FINAL RESULTS AGGREGATION, SAVING, AND PLOTTING
# ==============================================================================
print("\n" + "="*80)
print(" " * 20 + "COMPREHENSIVE COMPARISON OF ALL MODELS")
print("="*80 + "\n")

data_for_df = []
for model_name, results in all_results.items():
    data_for_df.append({
        "Model": model_name,
        "Accuracy": results.get('eval_accuracy', 0) * 100,
        "F1 Score": results.get('eval_f1', 0) * 100,
    })
    
results_df = pd.DataFrame(data_for_df)
print(results_df.to_string(index=False))
results_df.to_csv("final_results.csv", index=False)
print("\nResults saved to final_results.csv")

# Plotting performance comparison
fig, ax = plt.subplots(figsize=(12, 8))
results_df.set_index('Model').plot(kind='bar', ax=ax, rot=15)
ax.set_ylabel("Score (0-100)")
ax.set_title("Model Performance Comparison")
ax.grid(axis='y', linestyle='--', alpha=0.7)
ax.set_ylim(0, 100)
plt.tight_layout()
plt.savefig("results_comparison.png")
print("Plot saved to results_comparison.png")

# NEW: Plotting all training histories in a single figure
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.flatten()
fig.suptitle('Training and Validation Loss Across Models', fontsize=16)

for i, (model_name, trainer) in enumerate(all_trainers.items()):
    history = trainer.state.log_history
    
    train_metrics = pd.DataFrame([h for h in history if 'loss' in h])
    eval_metrics = pd.DataFrame([h for h in history if 'eval_loss' in h])
    
    # Plot Training Loss
    axes[i].plot(train_metrics['step'], train_metrics['loss'], label='Training Loss')
    # Plot Validation Loss on a secondary y-axis if needed, but plotting on same is fine
    axes[i].plot(eval_metrics['step'], eval_metrics['eval_loss'], label='Validation Loss', marker='o', linestyle='--')
    
    axes[i].set_title(model_name)
    axes[i].set_xlabel('Steps')
    axes[i].set_ylabel('Loss')
    axes[i].grid(True)
    axes[i].legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("all_training_histories.png")
print("Combined training history plot saved to all_training_histories.png")
plt.show()